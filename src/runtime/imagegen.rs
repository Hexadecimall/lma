//! Local, offline Stable Diffusion 1.5 text-to-image inference.
//!
//! The Diffusers directory must contain model_index.json, text_encoder/config.json,
//! text_encoder/model.safetensors, unet/config.json,
//! unet/diffusion_pytorch_model.safetensors, vae/config.json,
//! vae/diffusion_pytorch_model.safetensors, scheduler/scheduler_config.json and
//! tokenizer/tokenizer.json (or an explicit local tokenizer override).
//! Only the standard SD1.5 architecture and epsilon noise schedule are supported.
//! No subprocesses, downloads, pickle checkpoints, GGUF, SDXL or Flux are used.
//!
//! Candle's model builders memory-map the safetensors files. The caller must keep
//! those files immutable (no writes or truncation, including through other
//! processes) for the entire invocation. Read-only mappings do not enforce this
//! against other writers. Do not update a model directory while it is in use.

#[cfg(not(feature = "image-gen"))]
pub fn run(_config: crate::config::ImageConfig) -> Result<(), String> {
    Err("image generation is not compiled in; rebuild with --features image-gen".into())
}

#[cfg(feature = "image-gen")]
pub fn run(config: crate::config::ImageConfig) -> Result<(), String> {
    enabled::run(config)
}

#[cfg(feature = "image-gen")]
mod enabled {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use candle_core::{DType, Device, IndexOp, Module, Tensor};
    use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
    use image::{DynamicImage, ImageFormat, RgbImage};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    use serde_json::{Value, json};
    use tokenizers::Tokenizer;

    use crate::config::ImageConfig;

    const BOS: u32 = 49406;
    const EOS: u32 = 49407;
    const CONTEXT: usize = 77;
    const ASSETS: &[&str] = &[
        "model_index.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ];

    pub(super) fn run(config: ImageConfig) -> Result<(), String> {
        validate_settings(&config)?;
        let (root, tokenizer_path, inputs) = assets(&config)?;
        validate_architecture(&root)?;
        preflight_weights(&root)?;
        let output = prepare_output(&config.output, &root, &inputs)?;
        let tokenizer = load_tokenizer(&tokenizer_path)?;
        let prompt = encode_prompt(&tokenizer, &config.prompt, "prompt")?;
        let negative = encode_prompt(&tokenizer, &config.negative_prompt, "negative prompt")?;
        let device = select_device(&config.device)?;
        // CPU kernels require f32. Half-precision UNet weights/activations keep
        // the ordinary GPU path practical; CLIP and VAE remain f32 for stability.
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        eprintln!(
            "image-gen: SD1.5, {device:?}, {dtype:?}, {}x{}, {} DDIM steps, seed {}",
            config.width, config.height, config.steps, config.seed
        );
        eprintln!("image-gen: local weights must remain unchanged while generation is running");
        let result = generate(&config, &root, &prompt, &negative, &device, dtype)
            .map_err(|e| format!("SD1.5 inference failed: {e}"));
        let was_cpu = device.is_cpu();
        // GPU variants own allocations; a CPU-only Candle Device has no Drop.
        // Release the accelerator before a CPU retry in builds that include it.
        #[allow(clippy::drop_non_drop)]
        drop(device);
        let image = retry_on_cpu(&config.device, was_cpu, result, || {
            generate(&config, &root, &prompt, &negative, &Device::Cpu, DType::F32)
                .map_err(|e| format!("SD1.5 CPU inference failed: {e}"))
        })?;
        save_png(&image, &output)?;
        eprintln!("image-gen: saved {}", output.display());
        Ok(())
    }

    fn validate_settings(c: &ImageConfig) -> Result<(), String> {
        if c.prompt.trim().is_empty() {
            return Err("image prompt must not be empty".into());
        }
        if !(64..=1024).contains(&c.width)
            || !(64..=1024).contains(&c.height)
            || !c.width.is_multiple_of(8)
            || !c.height.is_multiple_of(8)
            || u64::from(c.width) * u64::from(c.height) > 1024 * 1024
        {
            return Err(
                "image dimensions must be 64..=1024, divisible by 8, and at most 1048576 pixels"
                    .into(),
            );
        }
        if !(1..=150).contains(&c.steps) {
            return Err("image steps must be in 1..=150".into());
        }
        if !c.guidance.is_finite() || !(1.0..=20.0).contains(&c.guidance) {
            return Err("image guidance must be finite and in 1..=20".into());
        }
        if !matches!(c.device.as_str(), "auto" | "cpu" | "metal" | "cuda") {
            return Err("image device must be auto, cpu, metal, or cuda".into());
        }
        if c.attention_slicing.is_some() {
            // Candle 0.11 stacks [slice, sequence, dim] results into rank 4,
            // then calls reshape_batch_dim_to_heads, which requires rank 3.
            // Even exact divisors fail; do not load gigabytes before reporting it.
            return Err(
                "attention slicing is unavailable with Candle 0.11 (upstream tensor-rank bug); omit -a and reduce image dimensions to lower memory use"
                    .into(),
            );
        }
        Ok(())
    }

    fn assets(c: &ImageConfig) -> Result<(PathBuf, PathBuf, Vec<PathBuf>), String> {
        if !c.model_dir.is_dir() {
            return Err(format!(
                "{} is not a local Diffusers SD1.5 directory; single-file checkpoints, GGUF, SDXL and Flux are unsupported. Supply a local SD1.5 Safetensors Diffusers folder with text_encoder/, unet/, vae/, scheduler/ and a CLIP tokenizer.json (or -t FILE). Nothing is downloaded automatically.",
                c.model_dir.display()
            ));
        }
        let root = fs::canonicalize(&c.model_dir).map_err(|e| format!("model directory: {e}"))?;
        let tokenizer = c
            .tokenizer
            .clone()
            .unwrap_or_else(|| root.join("tokenizer/tokenizer.json"));
        let mut inputs: Vec<_> = ASSETS.iter().map(|p| root.join(p)).collect();
        inputs.push(tokenizer.clone());
        let missing: Vec<_> = inputs
            .iter()
            .filter(|p| !p.is_file())
            .map(|p| p.display().to_string())
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "missing local SD1.5 assets:\n  {}\nRequired directory files:\n  {}\n  tokenizer/tokenizer.json (or -t FILE)\nOriginal Diffusers SD1.5 exports often lack tokenizer.json. Supply a separately obtained local openai/clip-vit-large-patch14 tokenizer.json via -t FILE. Nothing is downloaded automatically.",
                missing.join("\n  "),
                ASSETS.join("\n  ")
            ));
        }
        let inputs = inputs
            .iter()
            .map(|p| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display())))
            .collect::<Result<_, _>>()?;
        Ok((root, tokenizer, inputs))
    }

    fn read_config(root: &Path, relative: &str) -> Result<Value, String> {
        let path = root.join(relative);
        let file = fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.metadata().map_err(|e| e.to_string())?.len() > 1024 * 1024 {
            return Err(format!(
                "{}: architecture config exceeds 1 MiB",
                path.display()
            ));
        }
        let value: Value =
            serde_json::from_reader(file).map_err(|e| format!("{}: {e}", path.display()))?;
        if !value.is_object() {
            return Err(format!("{} must be a JSON object", path.display()));
        }
        Ok(value)
    }

    // Compare numeric JSON values by value (Diffusers writes both 1 and 1.0).
    fn same_value(actual: &Value, expected: &Value) -> bool {
        if actual.is_number() && expected.is_number() {
            actual.as_f64() == expected.as_f64()
        } else {
            actual == expected
        }
    }

    fn check_fields(
        value: &Value,
        name: &str,
        expected: Value,
        required: bool,
    ) -> Result<(), String> {
        for (key, expected) in expected.as_object().expect("static expected object") {
            match value.get(key) {
                Some(actual) if same_value(actual, expected) => {}
                None if !required => {}
                actual => {
                    return Err(format!(
                        "unsupported SD1.5 config {name}: {key} must be {expected}, found {}; SDXL, Flux, SD2, inpainting and custom architectures are not supported",
                        actual.map_or_else(|| "<missing>".into(), Value::to_string)
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_architecture(root: &Path) -> Result<(), String> {
        let index = read_config(root, "model_index.json")?;
        check_fields(
            &index,
            "model_index.json",
            json!({
                "_class_name": "StableDiffusionPipeline",
                "text_encoder": ["transformers", "CLIPTextModel"],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"]
            }),
            true,
        )?;
        for key in ["text_encoder_2", "tokenizer_2", "transformer"] {
            if index.get(key).is_some_and(|v| !v.is_null()) {
                return Err(format!(
                    "unsupported pipeline component {key}: only SD1.5 is supported"
                ));
            }
        }
        let text = read_config(root, "text_encoder/config.json")?;
        check_fields(
            &text,
            "text_encoder/config.json",
            json!({
                "architectures": ["CLIPTextModel"], "model_type": "clip_text_model",
                "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
                "num_attention_heads": 12, "max_position_embeddings": 77,
                "vocab_size": 49408, "hidden_act": "quick_gelu", "layer_norm_eps": 0.00001
            }),
            true,
        )?;
        // Historical SD1.5 CLIP configs have stale BOS/EOS/PAD ids (0/2/1).
        // Actual CLIP token ids come from tokenizer.json, not those config fields.
        let unet = read_config(root, "unet/config.json")?;
        check_fields(
            &unet,
            "unet/config.json",
            json!({
                "_class_name": "UNet2DConditionModel", "in_channels": 4, "out_channels": 4,
                "block_out_channels": [320, 640, 1280, 1280], "layers_per_block": 2,
                "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                "cross_attention_dim": 768, "attention_head_dim": 8, "act_fn": "silu",
                "norm_num_groups": 32, "norm_eps": 0.00001, "downsample_padding": 1,
                "center_input_sample": false, "flip_sin_to_cos": true,
                "freq_shift": 0, "mid_block_scale_factor": 1
            }),
            true,
        )?;
        check_fields(
            &unet,
            "unet/config.json",
            json!({
                "use_linear_projection": false, "only_cross_attention": false,
                "dual_cross_attention": false, "upcast_attention": false,
                "mid_block_type": "UNetMidBlock2DCrossAttn", "transformer_layers_per_block": 1,
                "class_embed_type": null, "addition_embed_type": null, "num_class_embeds": null,
                "time_embedding_type": "positional", "time_embedding_dim": null,
                "time_embedding_act_fn": null, "timestep_post_act": null, "time_cond_proj_dim": null,
                "resnet_time_scale_shift": "default", "resnet_skip_time_act": false,
                "resnet_out_scale_factor": 1, "class_embeddings_concat": false,
                "encoder_hid_dim": null, "encoder_hid_dim_type": null,
                "attention_type": "default", "cross_attention_norm": null,
                "num_attention_heads": null, "conv_in_kernel": 3, "conv_out_kernel": 3,
                "reverse_transformer_layers_per_block": null, "dropout": 0
            }),
            false,
        )?;
        let vae = read_config(root, "vae/config.json")?;
        check_fields(
            &vae,
            "vae/config.json",
            json!({
                "_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3,
                "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2,
                "latent_channels": 4, "norm_num_groups": 32, "act_fn": "silu",
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
            }),
            true,
        )?;
        check_fields(
            &vae,
            "vae/config.json",
            json!({
                "scaling_factor": 0.18215, "shift_factor": null,
                "latents_mean": null, "latents_std": null, "mid_block_add_attention": true,
                "use_quant_conv": true, "use_post_quant_conv": true
            }),
            false,
        )?;
        let scheduler = read_config(root, "scheduler/scheduler_config.json")?;
        check_fields(
            &scheduler,
            "scheduler/scheduler_config.json",
            json!({
                "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
                "num_train_timesteps": 1000, "steps_offset": 1
            }),
            true,
        )?;
        check_fields(
            &scheduler,
            "scheduler/scheduler_config.json",
            json!({
                "prediction_type": "epsilon", "trained_betas": null,
                "clip_sample": false, "set_alpha_to_one": false,
                "timestep_spacing": "leading", "rescale_betas_zero_snr": false,
                "thresholding": false
            }),
            false,
        )?;
        // PNDM is the original export's scheduler. We deliberately use DDIM,
        // retaining its training schedule, epsilon prediction and offset.
        match scheduler.get("_class_name").and_then(Value::as_str) {
            Some("PNDMScheduler" | "DDIMScheduler") => Ok(()),
            other => Err(format!(
                "unsupported scheduler {other:?}: expected SD1.5 PNDM or DDIM config (inference always uses DDIM eta=0)"
            )),
        }
    }

    fn preflight_weights(root: &Path) -> Result<(), String> {
        for (relative, expected) in [
            (
                "text_encoder/model.safetensors",
                vec![
                    (
                        "text_model.embeddings.token_embedding.weight",
                        vec![49408, 768],
                    ),
                    (
                        "text_model.embeddings.position_embedding.weight",
                        vec![77, 768],
                    ),
                ],
            ),
            (
                "unet/diffusion_pytorch_model.safetensors",
                vec![
                    ("conv_in.weight", vec![320, 4, 3, 3]),
                    ("conv_out.weight", vec![4, 320, 3, 3]),
                ],
            ),
            (
                "vae/diffusion_pytorch_model.safetensors",
                vec![
                    ("post_quant_conv.weight", vec![4, 4, 1, 1]),
                    ("decoder.conv_out.weight", vec![3, 128, 3, 3]),
                ],
            ),
        ] {
            check_weights(&root.join(relative), &expected)?;
        }
        Ok(())
    }

    fn check_weights(path: &Path, expected: &[(&str, Vec<usize>)]) -> Result<(), String> {
        // Only inspect headers: no tensor uploads or whole-file reads. This uses
        // the same file-immutability contract as Candle's model builders.
        let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(|e| format!(
                "invalid SD1.5 Safetensors file {}: {e}; expected actual local weights, not a Git LFS pointer, pickle checkpoint or incomplete download",
                path.display()
            ))?;
        for (name, shape) in expected {
            let tensor = weights
                .get(name)
                .map_err(|e| format!("{}: missing SD1.5 tensor {name}: {e}", path.display()))?;
            if tensor.shape() != shape {
                return Err(format!(
                    "{}: SD1.5 tensor {name} must have shape {shape:?}, found {:?}",
                    path.display(),
                    tensor.shape()
                ));
            }
        }
        for (name, tensor) in weights.tensors() {
            // Older CLIP exports can also contain integer position_ids buffers.
            // Only learned parameters must be floating point.
            if !["weight", "bias"]
                .iter()
                .any(|suffix| name == *suffix || name.ends_with(&format!(".{suffix}")))
            {
                continue;
            }
            if !matches!(
                DType::try_from(tensor.dtype()),
                Ok(DType::F16 | DType::BF16 | DType::F32)
            ) {
                return Err(format!(
                    "{}: unsupported SD1.5 weight dtype {:?} for {name}; expected F16, BF16 or F32",
                    path.display(),
                    tensor.dtype()
                ));
            }
        }
        Ok(())
    }

    fn load_tokenizer(path: &Path) -> Result<Tokenizer, String> {
        let mut tokenizer = Tokenizer::from_file(path).map_err(|e| {
            format!(
                "tokenizer {}: {e}; supply a local CLIP tokenizer.json with -t FILE",
                path.display()
            )
        })?;
        if tokenizer.token_to_id("<|startoftext|>") != Some(BOS)
            || tokenizer.token_to_id("<|endoftext|>") != Some(EOS)
            || tokenizer.get_vocab_size(true) != 49408
            || !matches!(
                tokenizer.get_model(),
                tokenizers::models::ModelWrapper::BPE(_)
            )
        {
            return Err("tokenizer must be the 49408-token CLIP BPE tokenizer (openai/clip-vit-large-patch14), with BOS=49406 and EOS=49407".into());
        }
        // Do not let saved tokenizer settings silently truncate or pre-pad.
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        Ok(tokenizer)
    }

    fn encode_prompt(tokenizer: &Tokenizer, text: &str, label: &str) -> Result<Vec<u32>, String> {
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| format!("{label} tokenization: {e}"))?;
        pad_tokens(encoding.get_ids(), label)
    }

    fn pad_tokens(content: &[u32], label: &str) -> Result<Vec<u32>, String> {
        if content.len() > CONTEXT - 2 {
            return Err(format!(
                "{label} is too long: {} CLIP tokens; at most 75 content tokens (77 including BOS/EOS) are supported; shorten it",
                content.len()
            ));
        }
        if content.iter().any(|&id| id >= BOS) {
            return Err(format!(
                "{label} contains a reserved CLIP special token or an out-of-vocabulary id"
            ));
        }
        let mut tokens = Vec::with_capacity(CONTEXT);
        tokens.push(BOS);
        tokens.extend_from_slice(content);
        // SD1.5 pads with EOS, not the stale pad_token_id in CLIP config.json.
        tokens.resize(CONTEXT, EOS);
        Ok(tokens)
    }

    fn select_device(name: &str) -> Result<Device, String> {
        match name {
            "cpu" => Ok(Device::Cpu),
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| format!("cannot initialize CUDA device 0: {e}"))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("CUDA image generation requires rebuilding with --features cuda".into())
                }
            }
            "metal" => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0)
                        .map_err(|e| format!("cannot initialize Metal device 0: {e}"))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal image generation requires rebuilding with --features metal".into())
                }
            }
            "auto" => {
                #[cfg(feature = "cuda")]
                match Device::new_cuda(0) {
                    Ok(device) => return Ok(device),
                    Err(e) => eprintln!("image-gen: CUDA unavailable: {e}"),
                }
                #[cfg(feature = "metal")]
                match Device::new_metal(0) {
                    Ok(device) => return Ok(device),
                    Err(e) => eprintln!("image-gen: Metal unavailable: {e}"),
                }
                Ok(Device::Cpu)
            }
            _ => Err("image device must be auto, cpu, metal, or cuda".into()),
        }
    }

    fn retry_on_cpu<T>(
        requested: &str,
        was_cpu: bool,
        result: Result<T, String>,
        retry: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        match result {
            Err(accelerator_error) if requested == "auto" && !was_cpu => {
                eprintln!(
                    "image-gen: accelerator failed: {accelerator_error}; retrying once on CPU (F32)"
                );
                retry().map_err(|cpu_error| {
                    format!("{accelerator_error}\nAutomatic CPU fallback also failed: {cpu_error}")
                })
            }
            result => result,
        }
    }

    fn initial_noise(seed: u64, count: usize) -> Vec<f32> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect()
    }

    fn generate(
        c: &ImageConfig,
        root: &Path,
        prompt: &[u32],
        negative: &[u32],
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<RgbImage> {
        let sd = StableDiffusionConfig::v1_5(
            c.attention_slicing,
            Some(c.height as usize),
            Some(c.width as usize),
        );
        let guided = c.guidance > 1.0;
        let mut scheduler = sd.build_scheduler(c.steps)?; // DDIM defaults to eta=0.
        eprintln!("image-gen: encoding prompt with CLIP");
        let embeddings = {
            // These safe Candle builders internally use mmap, not whole-file
            // buffers. The file-immutability contract is documented above.
            let clip = stable_diffusion::build_clip_transformer(
                &sd.clip,
                root.join("text_encoder/model.safetensors"),
                device,
                DType::F32,
            )?;
            let tokens = if guided {
                [negative, prompt].concat()
            } else {
                prompt.to_vec()
            };
            let tokens = Tensor::from_vec(tokens, (if guided { 2 } else { 1 }, CONTEXT), device)?;
            clip.forward(&tokens)?.to_dtype(dtype)?
        }; // Release CLIP weights before loading UNet.
        let height = c.height as usize / 8;
        let width = c.width as usize / 8;
        // Explicit CPU RNG works even where Device::set_seed is unsupported.
        // Seeds reproduce initial noise; GPU results need not be bit-identical.
        let noise = initial_noise(c.seed, 4 * height * width);
        let mut latents = (Tensor::from_vec(noise, (1, 4, height, width), device)?
            * scheduler.init_noise_sigma())?
        .to_dtype(dtype)?;
        eprintln!("image-gen: loading UNet");
        let unet = sd.build_unet(
            root.join("unet/diffusion_pytorch_model.safetensors"),
            device,
            4,
            false,
            dtype,
        )?;
        let timesteps = scheduler.timesteps().to_vec();
        let interval = c.steps.div_ceil(20);
        let start = Instant::now();
        for (index, &timestep) in timesteps.iter().enumerate() {
            let input = if guided {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            let input = scheduler.scale_model_input(input, timestep)?;
            let prediction = unet.forward(&input, timestep as f64, &embeddings)?;
            let prediction = if guided {
                let unconditional = prediction.narrow(0, 0, 1)?;
                let conditional = prediction.narrow(0, 1, 1)?;
                (&unconditional + ((conditional - &unconditional)? * c.guidance)?)?
            } else {
                prediction
            };
            latents = scheduler.step(&prediction, timestep, &latents)?;
            if index == 0 || (index + 1) % interval == 0 || index + 1 == timesteps.len() {
                eprintln!(
                    "image-gen: step {}/{} ({:.1}s elapsed)",
                    index + 1,
                    timesteps.len(),
                    start.elapsed().as_secs_f32()
                );
            }
        }
        drop(unet);
        drop(embeddings);
        eprintln!("image-gen: decoding RGB image with VAE");
        let vae = sd.build_vae(
            root.join("vae/diffusion_pytorch_model.safetensors"),
            device,
            DType::F32,
        )?;
        let decoded = vae.decode(&(latents.to_dtype(DType::F32)? / 0.18215)?)?;
        decoded_to_rgb(&decoded, c.width, c.height)
    }

    fn decoded_to_rgb(decoded: &Tensor, width: u32, height: u32) -> candle_core::Result<RgbImage> {
        if decoded.dims() != [1, 3, height as usize, width as usize] {
            candle_core::bail!("unexpected decoded image dimensions: {:?}", decoded.dims());
        }
        let pixels = decoded
            .i(0)?
            .permute((1, 2, 0))?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if pixels.iter().any(|v| !v.is_finite()) {
            candle_core::bail!("VAE produced non-finite pixels; refusing to save a corrupt image");
        }
        let bytes = pixels
            .into_iter()
            .map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        RgbImage::from_raw(width, height, bytes)
            .ok_or_else(|| candle_core::Error::Msg("invalid RGB image buffer".into()))
    }

    fn refuse_existing(path: &Path) -> Result<(), String> {
        match fs::symlink_metadata(path) {
            Ok(_) => Err(format!(
                "refusing to overwrite existing output {}",
                path.display()
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("cannot inspect output {}: {e}", path.display())),
        }
    }

    fn prepare_output(output: &Path, root: &Path, inputs: &[PathBuf]) -> Result<PathBuf, String> {
        let name = output.file_name().ok_or("output must name a PNG file")?;
        if !output
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        {
            return Err("image output must have a .png extension".into());
        }
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = fs::canonicalize(parent).map_err(|e| {
            format!(
                "output directory {} must already exist: {e}",
                parent.display()
            )
        })?;
        if !parent.is_dir() {
            return Err("output parent must be a directory".into());
        }
        let target = parent.join(name);
        if target.starts_with(root)
            || inputs
                .iter()
                .any(|p| p == &target || p.starts_with(&target))
        {
            return Err("output must not intersect the model directory or any input asset".into());
        }
        refuse_existing(&target)?;
        Ok(target)
    }

    fn save_png(image: &RgbImage, output: &Path) -> Result<(), String> {
        refuse_existing(output)?;
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut temporary = tempfile::Builder::new()
            .prefix(".lma-image-")
            .suffix(".png")
            .tempfile_in(parent)
            .map_err(|e| format!("cannot create temporary PNG in {}: {e}", parent.display()))?;
        DynamicImage::ImageRgb8(image.clone())
            .write_to(temporary.as_file_mut(), ImageFormat::Png)
            .map_err(|e| format!("PNG encoding failed: {e}"))?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|e| format!("cannot flush PNG: {e}"))?;
        // Same-directory publication: no partial final PNG, and no replacement
        // even if another writer creates the destination during inference.
        temporary.persist_noclobber(output).map_err(|e| {
            format!(
                "cannot publish {} without overwriting: {}",
                output.display(),
                e.error
            )
        })?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn config() -> ImageConfig {
            ImageConfig {
                model_dir: PathBuf::from("model"),
                tokenizer: None,
                prompt: "a candle".into(),
                negative_prompt: String::new(),
                output: PathBuf::from("image.png"),
                width: 512,
                height: 512,
                steps: 20,
                seed: 42,
                guidance: 7.5,
                device: "cpu".into(),
                attention_slicing: None,
            }
        }

        #[test]
        fn validates_bounded_settings() {
            let mut c = config();
            assert!(validate_settings(&c).is_ok());
            for width in [0, 63, 65, 1025, u32::MAX] {
                c.width = width;
                assert!(validate_settings(&c).is_err());
            }
            c.width = 512;
            for steps in [0, 151, usize::MAX] {
                c.steps = steps;
                assert!(validate_settings(&c).is_err());
            }
            c.steps = 1;
            for guidance in [f64::NAN, f64::INFINITY, 0.9, 20.1] {
                c.guidance = guidance;
                assert!(validate_settings(&c).is_err());
            }
            c.guidance = 7.5;
            for size in [0, 3, 17, usize::MAX] {
                c.attention_slicing = Some(size);
                assert!(validate_settings(&c).is_err());
            }
            c.attention_slicing = None;
            assert!(validate_settings(&c).is_ok());
            c.prompt = " \n".into();
            assert!(validate_settings(&c).is_err());
        }

        #[test]
        fn broken_candle_slicing_is_rejected_before_loading_weights() {
            let mut c = config();
            for guidance in [1., 7.5] {
                c.guidance = guidance;
                for size in [0, 1, 2, 4, 8, 16, usize::MAX] {
                    c.attention_slicing = Some(size);
                    let error = validate_settings(&c).unwrap_err();
                    assert!(error.contains("Candle 0.11"));
                    assert!(error.contains("omit -a"));
                }
            }
        }

        #[test]
        fn automatic_inference_fallback_retries_once_and_preserves_errors() {
            let retried = std::cell::Cell::new(0);
            let result = retry_on_cpu("auto", false, Err("GPU failure".into()), || {
                retried.set(retried.get() + 1);
                Ok(42)
            });
            assert_eq!(result.unwrap(), 42);
            assert_eq!(retried.get(), 1);
            let error = retry_on_cpu::<()>("auto", false, Err("GPU failure".into()), || {
                Err("CPU failure".into())
            })
            .unwrap_err();
            assert!(error.contains("GPU failure"));
            assert!(error.contains("CPU failure"));
            for (requested, was_cpu) in [
                ("cpu", true),
                ("auto", true),
                ("metal", false),
                ("cuda", false),
            ] {
                let error = retry_on_cpu::<()>(requested, was_cpu, Err("original".into()), || {
                    panic!("explicit devices and an initial CPU attempt must not retry")
                })
                .unwrap_err();
                assert_eq!(error, "original");
            }
            assert_eq!(
                retry_on_cpu("auto", false, Ok(7), || panic!("success must not retry")).unwrap(),
                7
            );
        }

        #[test]
        fn device_selection_obeys_explicit_cpu_and_uncompiled_backends() {
            assert!(select_device("cpu").unwrap().is_cpu());
            assert!(select_device("invalid").is_err());
            #[cfg(not(feature = "metal"))]
            assert!(
                select_device("metal")
                    .unwrap_err()
                    .contains("--features metal")
            );
            #[cfg(not(feature = "cuda"))]
            assert!(
                select_device("cuda")
                    .unwrap_err()
                    .contains("--features cuda")
            );
            #[cfg(not(any(feature = "metal", feature = "cuda")))]
            assert!(select_device("auto").unwrap().is_cpu());
        }

        #[test]
        fn safetensors_preflight_checks_shapes_dtypes_and_actual_payload() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("synthetic.safetensors");
            let expected = [("weight", vec![2, 3])];
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                candle_core::safetensors::save(
                    &std::collections::HashMap::from([
                        ("weight", tensor),
                        (
                            "text_model.embeddings.position_ids",
                            Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap(),
                        ),
                    ]),
                    &path,
                )
                .unwrap();
                check_weights(&path, &expected).unwrap();
                // Exercise Candle's serialized dtype conversion, not just JSON.
                let loaded = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
                assert_eq!(loaded["weight"].dtype(), dtype);
                assert_eq!(
                    loaded["weight"]
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap(),
                    vec![vec![1., 2., 3.], vec![4., 5., 6.]]
                );
                assert!(
                    check_weights(&path, &[("weight", vec![3, 2])])
                        .unwrap_err()
                        .contains("shape")
                );
                assert!(
                    check_weights(&path, &[("missing", vec![2, 3])])
                        .unwrap_err()
                        .contains("missing SD1.5 tensor")
                );
            }
            let integer = Tensor::zeros((2, 3), DType::U8, &Device::Cpu).unwrap();
            candle_core::safetensors::save(
                &std::collections::HashMap::from([("weight", integer)]),
                &path,
            )
            .unwrap();
            assert!(
                check_weights(&path, &expected)
                    .unwrap_err()
                    .contains("unsupported SD1.5 weight dtype")
            );
            for bytes in [
                b"version https://git-lfs.github.com/spec/v1\n".as_slice(),
                &[0u8; 8],
            ] {
                fs::write(&path, bytes).unwrap();
                let error = check_weights(&path, &expected).unwrap_err();
                assert!(error.contains("actual local weights"));
                assert!(error.contains("synthetic.safetensors"));
            }
        }

        #[test]
        fn ddim_matches_sd15_epsilon_reference_including_final_step() {
            // Independent scalar reference: scaled-linear training betas,
            // leading timesteps with offset 1, eta=0, set_alpha_to_one=false.
            let mut alphas = Vec::with_capacity(1000);
            let mut product = 1f64;
            for index in 0..1000 {
                let beta = (0.00085f64.sqrt()
                    + (0.012f64.sqrt() - 0.00085f64.sqrt()) * index as f64 / 999.)
                    .powi(2);
                product *= 1. - beta;
                alphas.push(product);
            }
            for steps in [1, 2, 20, 150] {
                let sd = StableDiffusionConfig::v1_5(None, Some(64), Some(64));
                let mut scheduler = sd.build_scheduler(steps).unwrap();
                assert_eq!(scheduler.init_noise_sigma(), 1.);
                let ratio = 1000 / steps;
                let timesteps: Vec<_> = (0..steps).map(|i| i * ratio + 1).rev().collect();
                assert_eq!(scheduler.timesteps(), timesteps);
                let mut sample = Tensor::new(&[0.25f32, -0.5], &Device::Cpu).unwrap();
                let prediction = Tensor::new(&[0.1f32, -0.2], &Device::Cpu).unwrap();
                let mut expected = [0.25f64, -0.5];
                for timestep in timesteps {
                    let scaled = scheduler
                        .scale_model_input(sample.clone(), timestep)
                        .unwrap();
                    assert_eq!(
                        scaled.to_vec1::<f32>().unwrap(),
                        sample.to_vec1::<f32>().unwrap()
                    );
                    let alpha = alphas[timestep];
                    let previous = alphas[timestep.saturating_sub(ratio)];
                    for (value, epsilon) in expected.iter_mut().zip([0.1f32, -0.2]) {
                        let epsilon = f64::from(epsilon);
                        let original = (*value - (1. - alpha).sqrt() * epsilon) / alpha.sqrt();
                        *value = previous.sqrt() * original + (1. - previous).sqrt() * epsilon;
                    }
                    sample = scheduler.step(&prediction, timestep, &sample).unwrap();
                    assert_eq!(sample.dtype(), DType::F32);
                    for (actual, reference) in sample.to_vec1::<f32>().unwrap().iter().zip(expected)
                    {
                        assert!(
                            (f64::from(*actual) - reference).abs() < 0.0001,
                            "{steps} steps, timestep {timestep}: {actual} != {reference}"
                        );
                    }
                }
            }
        }

        #[test]
        fn clip_bos_eos_padding_and_limits() {
            let empty = pad_tokens(&[], "negative prompt").unwrap();
            assert_eq!(empty.len(), 77);
            assert_eq!(empty[0], BOS);
            assert!(empty[1..].iter().all(|&id| id == EOS));
            let full = pad_tokens(&[10; 75], "prompt").unwrap();
            assert_eq!(full[0], BOS);
            assert_eq!(&full[1..76], &[10; 75]);
            assert_eq!(full[76], EOS);
            assert!(
                pad_tokens(&[10; 76], "prompt")
                    .unwrap_err()
                    .contains("too long")
            );
            assert!(pad_tokens(&[BOS], "prompt").is_err());
            assert!(pad_tokens(&[49408], "prompt").is_err());
        }

        #[test]
        fn prompt_tokenization_does_not_add_special_tokens_twice() {
            let model = tokenizers::models::wordlevel::WordLevel::builder()
                .vocab([("candle".to_owned(), 10)].into_iter().collect())
                .build()
                .unwrap();
            let tokenizer = Tokenizer::new(model);
            let tokens = encode_prompt(&tokenizer, "candle", "prompt").unwrap();
            assert_eq!(&tokens[..3], &[BOS, 10, EOS]);
            assert_eq!(tokens.len(), CONTEXT);
        }

        #[test]
        fn saved_tokenizer_cannot_silently_truncate_or_pad_prompts() {
            // Synthetic BPE vocabulary: no external tokenizer or model fixture.
            let mut vocab: tokenizers::models::bpe::Vocab =
                (0..BOS).map(|id| (format!("token{id}"), id)).collect();
            vocab.remove("token0");
            vocab.insert("a".into(), 0);
            vocab.insert("<|startoftext|>".into(), BOS);
            vocab.insert("<|endoftext|>".into(), EOS);
            let model = tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap();
            let mut tokenizer = Tokenizer::new(model);
            tokenizer.with_pre_tokenizer(Some(
                tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
            ));
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: 3,
                    ..Default::default()
                }))
                .unwrap();
            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(100),
                pad_id: EOS,
                pad_token: "<|endoftext|>".into(),
                ..Default::default()
            }));
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("tokenizer.json");
            tokenizer.save(&path, false).unwrap();
            let tokenizer = load_tokenizer(&path).unwrap();
            assert!(tokenizer.get_truncation().is_none());
            assert!(tokenizer.get_padding().is_none());
            let tokens = encode_prompt(&tokenizer, &"a ".repeat(75), "prompt").unwrap();
            assert_eq!(&tokens[1..76], &[0; 75]);
            assert!(
                encode_prompt(&tokenizer, &"a ".repeat(76), "negative prompt")
                    .unwrap_err()
                    .contains("negative prompt is too long")
            );
        }

        #[test]
        fn accepts_legacy_sd15_configs_and_rejects_changed_architectures() {
            let dir = tempfile::tempdir().unwrap();
            // Original Diffusers SD1.5 fields, including stale CLIP special ids
            // and PNDM's absent prediction_type/scaling_factor defaults.
            let configs = [
                (
                    "model_index.json",
                    json!({
                        "_class_name": "StableDiffusionPipeline",
                        "text_encoder": ["transformers", "CLIPTextModel"],
                        "unet": ["diffusers", "UNet2DConditionModel"],
                        "vae": ["diffusers", "AutoencoderKL"]
                    }),
                ),
                (
                    "text_encoder/config.json",
                    json!({
                        "architectures": ["CLIPTextModel"], "model_type": "clip_text_model",
                        "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
                        "num_attention_heads": 12, "max_position_embeddings": 77,
                        "vocab_size": 49408, "hidden_act": "quick_gelu", "layer_norm_eps": 1e-5,
                        "bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1
                    }),
                ),
                (
                    "unet/config.json",
                    json!({
                        "_class_name": "UNet2DConditionModel", "in_channels": 4, "out_channels": 4,
                        "block_out_channels": [320, 640, 1280, 1280], "layers_per_block": 2,
                        "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                        "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                        "cross_attention_dim": 768, "attention_head_dim": 8, "act_fn": "silu",
                        "norm_num_groups": 32, "norm_eps": 1e-5, "downsample_padding": 1,
                        "center_input_sample": false, "flip_sin_to_cos": true,
                        "freq_shift": 0, "mid_block_scale_factor": 1
                    }),
                ),
                (
                    "vae/config.json",
                    json!({
                        "_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3,
                        "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2,
                        "latent_channels": 4, "norm_num_groups": 32, "act_fn": "silu",
                        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
                    }),
                ),
                (
                    "scheduler/scheduler_config.json",
                    json!({
                        "_class_name": "PNDMScheduler", "beta_start": 0.00085, "beta_end": 0.012,
                        "beta_schedule": "scaled_linear", "num_train_timesteps": 1000,
                        "steps_offset": 1, "set_alpha_to_one": false, "clip_sample": false,
                        "skip_prk_steps": true, "trained_betas": null
                    }),
                ),
            ];
            for (relative, value) in &configs {
                let path = dir.path().join(relative);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
            }
            validate_architecture(dir.path()).unwrap();
            for (relative, field, replacement) in [
                (
                    "model_index.json",
                    "_class_name",
                    json!("StableDiffusionXLPipeline"),
                ),
                ("model_index.json", "_class_name", json!("FluxPipeline")),
                ("unet/config.json", "in_channels", json!(9)),
                ("unet/config.json", "cross_attention_dim", json!(1024)),
                ("vae/config.json", "scaling_factor", json!(0.13025)),
                (
                    "scheduler/scheduler_config.json",
                    "prediction_type",
                    json!("v_prediction"),
                ),
            ] {
                let original = &configs
                    .iter()
                    .find(|(name, _)| *name == relative)
                    .unwrap()
                    .1;
                let mut changed = original.clone();
                changed[field] = replacement;
                let path = dir.path().join(relative);
                fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
                assert!(
                    validate_architecture(dir.path()).is_err(),
                    "{relative}: {field}"
                );
                fs::write(path, serde_json::to_vec(original).unwrap()).unwrap();
            }
        }

        #[cfg(feature = "metal")]
        #[test]
        #[ignore = "requires a working Metal device; synthetic tensor test, not model inference"]
        fn metal_precision_boundaries_and_rgb_readback() {
            let device = Device::new_metal(0).expect("Metal device 0");
            let timestep = Tensor::new(&[981f32, 1.], &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let embedding = stable_diffusion::embeddings::Timesteps::new(320, true, 0.)
                .forward(&timestep)
                .unwrap();
            assert_eq!(embedding.dtype(), DType::F16);
            assert_eq!(embedding.dims(), &[2, 320]);
            assert!(
                embedding
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite())
            );

            let sd = StableDiffusionConfig::v1_5(None, Some(64), Some(64));
            let mut scheduler = sd.build_scheduler(2).unwrap();
            let sample = Tensor::new(&[0.25f32, -0.5], &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let prediction = Tensor::zeros(2, DType::F16, &device).unwrap();
            let timestep = scheduler.timesteps()[0];
            let scaled = scheduler
                .scale_model_input(sample.clone(), timestep)
                .unwrap();
            let stepped = scheduler.step(&prediction, timestep, &scaled).unwrap();
            assert_eq!(stepped.dtype(), DType::F16);
            assert!(
                stepped
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );

            let decoded = Tensor::from_vec(vec![-1f32, 0., 1.], (1, 3, 1, 1), &device).unwrap();
            assert_eq!(decoded.dtype(), DType::F32);
            assert_eq!(
                decoded_to_rgb(&decoded, 1, 1).unwrap().into_raw(),
                vec![0, 128, 255]
            );
            device.synchronize().unwrap();
        }

        #[test]
        fn missing_assets_explain_the_offline_folder_contract() {
            let dir = tempfile::tempdir().unwrap();
            let mut c = config();
            c.model_dir = dir.path().into();
            let error = assets(&c).unwrap_err();
            for relative in ASSETS {
                assert!(error.contains(relative));
            }
            assert!(error.contains("-t FILE"));
            assert!(error.contains("clip-vit-large-patch14"));
            assert!(error.contains("Nothing is downloaded"));
            c.model_dir = dir.path().join("not-installed");
            let error = assets(&c).unwrap_err();
            assert!(error.contains("local SD1.5 Safetensors Diffusers folder"));
            assert!(error.contains("-t FILE"));
            assert!(error.contains("Nothing is downloaded"));
        }

        #[test]
        fn rejects_incompatible_config_fields() {
            let expected = json!({"cross_attention_dim": 768});
            assert!(
                check_fields(
                    &json!({"cross_attention_dim": 2048}),
                    "unet",
                    expected.clone(),
                    true
                )
                .is_err()
            );
            assert!(check_fields(&json!({}), "unet", expected, true).is_err());
            assert!(same_value(&json!(1), &json!(1.0)));
            assert!(
                check_fields(
                    &json!({"prediction_type": "v_prediction"}),
                    "scheduler",
                    json!({"prediction_type": "epsilon"}),
                    false
                )
                .is_err()
            );
        }

        #[test]
        fn noise_is_seeded_without_device_rng() {
            let a = initial_noise(42, 100);
            assert_eq!(a, initial_noise(42, 100));
            assert_ne!(a, initial_noise(43, 100));
            assert!(a.iter().all(|v| v.is_finite()));
        }

        #[test]
        fn rgb_normalization_and_layout() {
            let decoded = Tensor::from_vec(
                vec![-1f32, 1., 0., -2., 2., -1.],
                (1, 3, 1, 2),
                &Device::Cpu,
            )
            .unwrap();
            let image = decoded_to_rgb(&decoded, 2, 1).unwrap();
            assert_eq!(image.into_raw(), vec![0, 128, 255, 255, 0, 0]);
            let invalid = Tensor::from_vec(vec![f32::NAN; 3], (1, 3, 1, 1), &Device::Cpu).unwrap();
            assert!(decoded_to_rgb(&invalid, 1, 1).is_err());
        }

        #[test]
        fn png_publication_never_clobbers_or_leaves_partial_output() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("tiny.png");
            let image = RgbImage::from_pixel(2, 2, image::Rgb([12, 34, 56]));
            save_png(&image, &path).unwrap();
            let original = fs::read(&path).unwrap();
            assert_eq!(image::load_from_memory(&original).unwrap().to_rgb8(), image);
            assert!(save_png(&RgbImage::new(1, 1), &path).is_err());
            assert_eq!(fs::read(&path).unwrap(), original);
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        }

        #[test]
        fn output_cannot_intersect_inputs_or_model_tree() {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("model");
            fs::create_dir(&root).unwrap();
            let root = fs::canonicalize(root).unwrap();
            assert!(prepare_output(&root.join("out.png"), &root, &[]).is_err());
            let external = fs::canonicalize(dir.path()).unwrap().join("tokenizer.png");
            assert!(prepare_output(&external, &root, std::slice::from_ref(&external)).is_err());
            let out = dir.path().join("out.png");
            assert!(prepare_output(&out, &root, &[]).is_ok());
            fs::write(&out, "existing").unwrap();
            assert!(prepare_output(&out, &root, &[]).is_err());
        }

        #[cfg(unix)]
        #[test]
        fn output_rejects_symlinks_and_model_directory_aliases() {
            use std::os::unix::fs::symlink;
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("model");
            fs::create_dir(&root).unwrap();
            let root = fs::canonicalize(root).unwrap();
            let alias = dir.path().join("alias");
            symlink(&root, &alias).unwrap();
            assert!(prepare_output(&alias.join("out.png"), &root, &[]).is_err());
            let dangling = dir.path().join("dangling.png");
            symlink(dir.path().join("absent"), &dangling).unwrap();
            assert!(prepare_output(&dangling, &root, &[]).is_err());
            assert!(save_png(&RgbImage::new(1, 1), &dangling).is_err());
        }
    }
}
