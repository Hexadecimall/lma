use super::{LANGUAGES, MAX_AUDIO_SECONDS, SAMPLE_RATE, Segment, Transcript, validate_samples};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as whisper, Config, model::Whisper};
use std::{io::Read, path::Path};
use tokenizers::Tokenizer;

#[path = "mel.rs"]
mod mel;

const MAX_WEIGHTS_BYTES: u64 = 4 * 1024 * 1024 * 1024;

pub(super) fn transcribe(
    directory: &Path,
    samples: &[f32],
    language: Option<&str>,
    threads: usize,
    gpu: bool,
) -> Result<Transcript, String> {
    validate_samples(samples, MAX_AUDIO_SECONDS)?;
    if !directory.is_dir() {
        return Err(format!(
            "Whisper model {} must be a local directory containing config.json, tokenizer.json, and model.safetensors; whisper.cpp GGML/GGUF files are not supported",
            directory.display()
        ));
    }
    let config: Config =
        serde_json::from_slice(&read_asset(&directory.join("config.json"), 1024 * 1024)?)
            .map_err(|e| format!("invalid Whisper config.json: {e}"))?;
    validate_config(&config)?;
    let tokenizer = Tokenizer::from_bytes(read_asset(
        &directory.join("tokenizer.json"),
        32 * 1024 * 1024,
    )?)
    .map_err(|e| format!("invalid Whisper tokenizer.json: {e}"))?;
    let tokens = Tokens::new(&tokenizer, config.vocab_size)?;
    let english_only = config.vocab_size == 51864;
    if english_only && language.is_some_and(|language| language != "en") {
        return Err("this English-only Whisper model supports only -l en or -l auto".into());
    }
    let mut detected_language = language
        .map(str::to_owned)
        .or_else(|| english_only.then(|| "en".into()));
    let mut language_token = if english_only {
        None
    } else {
        language
            .map(|code| token(&tokenizer, &format!("<|{code}|>"), config.vocab_size))
            .transpose()?
    };
    let device = device(gpu)?;
    // Buffered loading avoids mmap's requirement that external files remain immutable
    // while the desktop app is running. Only uncompressed single-file weights are accepted.
    let weights = read_asset(&directory.join("model.safetensors"), MAX_WEIGHTS_BYTES)?;
    let builder = VarBuilder::from_buffered_safetensors(weights, whisper::DTYPE, &device)
        .map_err(|e| format!("cannot read Whisper model.safetensors: {e}"))?;
    let mut model = Whisper::load(&builder, config.clone()).map_err(|e| {
        format!("cannot load Whisper weights (expected Hugging Face Whisper safetensors): {e}")
    })?;
    drop(builder);
    let filters = mel::filters(config.num_mel_bins);
    let mut segments = Vec::new();
    for (index, chunk) in samples.chunks(whisper::N_SAMPLES).enumerate() {
        // Avoid running the decoder on digital silence, a common source of hallucinations.
        if chunk.iter().all(|s| s.abs() < 1e-6) {
            continue;
        }
        let spectrogram = mel::spectrogram(chunk, &filters, config.num_mel_bins, threads);
        let text = infer_chunk(
            &mut model,
            &tokenizer,
            &tokens,
            &device,
            spectrogram,
            &mut language_token,
            &mut detected_language,
            english_only,
        )
        .map_err(|e| format!("Whisper inference failed for chunk {}: {e}", index + 1))?;
        if !text.is_empty() {
            let start = index * whisper::N_SAMPLES;
            segments.push(Segment {
                start_ms: (start as u64 * 1000) / SAMPLE_RATE as u64,
                end_ms: ((start + chunk.len()) as u64 * 1000) / SAMPLE_RATE as u64,
                text,
            });
        }
    }
    let text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(Transcript {
        text,
        language: detected_language,
        segments,
    })
}

fn read_asset(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let file = crate::local_file::open_regular(path).map_err(|e| {
        format!(
            "cannot open local Whisper asset {}: {e}; no models are downloaded automatically",
            path.display()
        )
    })?;
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > limit {
        return Err(format!(
            "Whisper asset {} must be a non-empty regular file of at most {limit} bytes",
            path.display()
        ));
    }
    let mut data = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut data)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if data.len() as u64 > limit {
        return Err(format!(
            "Whisper asset {} exceeds its size limit",
            path.display()
        ));
    }
    Ok(data)
}

fn validate_config(config: &Config) -> Result<(), String> {
    if !matches!(config.num_mel_bins, 80 | 128)
        || config.max_source_positions != 1500
        || !(8..=448).contains(&config.max_target_positions)
        || !(1..=32).contains(&config.encoder_layers)
        || !(1..=32).contains(&config.decoder_layers)
        || !(4..=1280).contains(&config.d_model)
        || !config.d_model.is_multiple_of(2)
        || !(1..=20).contains(&config.encoder_attention_heads)
        || !(1..=20).contains(&config.decoder_attention_heads)
        || !config
            .d_model
            .is_multiple_of(config.encoder_attention_heads)
        || !config
            .d_model
            .is_multiple_of(config.decoder_attention_heads)
        || !matches!(config.vocab_size, 51864..=51866)
        || config
            .suppress_tokens
            .iter()
            .any(|id| *id as usize >= config.vocab_size)
    {
        return Err("unsupported Whisper config: expected standard 80/128-mel Hugging Face Whisper dimensions and vocabulary".into());
    }
    Ok(())
}

fn device(gpu: bool) -> Result<Device, String> {
    if !gpu {
        return Ok(Device::Cpu);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        return Device::new_metal(0).map_err(|e| {
            format!("cannot initialize Whisper Metal device: {e}; use -g off for CPU")
        });
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        return Device::new_cuda(0).map_err(|e| {
            format!("cannot initialize Whisper CUDA device: {e}; use -g off for CPU")
        });
    }
    #[allow(unreachable_code)]
    Err("no supported Whisper GPU backend compiled; use -g off".into())
}

fn token(tokenizer: &Tokenizer, name: &str, vocab: usize) -> Result<u32, String> {
    tokenizer
        .token_to_id(name)
        .filter(|id| (*id as usize) < vocab)
        .ok_or_else(|| format!("Whisper tokenizer is missing supported token {name}"))
}

struct Tokens {
    sot: u32,
    eot: u32,
    transcribe: u32,
    no_timestamps: u32,
    no_speech: u32,
}
impl Tokens {
    fn new(tokenizer: &Tokenizer, vocab: usize) -> Result<Self, String> {
        Ok(Self {
            sot: token(tokenizer, whisper::SOT_TOKEN, vocab)?,
            eot: token(tokenizer, whisper::EOT_TOKEN, vocab)?,
            transcribe: token(tokenizer, whisper::TRANSCRIBE_TOKEN, vocab)?,
            no_timestamps: token(tokenizer, whisper::NO_TIMESTAMPS_TOKEN, vocab)?,
            no_speech: whisper::NO_SPEECH_TOKENS
                .iter()
                .find_map(|name| token(tokenizer, name, vocab).ok())
                .ok_or("Whisper tokenizer is missing a no-speech token")?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_chunk(
    model: &mut Whisper,
    tokenizer: &Tokenizer,
    special: &Tokens,
    device: &Device,
    spectrogram: Vec<f32>,
    language_token: &mut Option<u32>,
    language: &mut Option<String>,
    english_only: bool,
) -> Result<String, String> {
    infer_inner(
        model,
        tokenizer,
        special,
        device,
        spectrogram,
        language_token,
        language,
        english_only,
    )
    .map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn infer_inner(
    model: &mut Whisper,
    tokenizer: &Tokenizer,
    special: &Tokens,
    device: &Device,
    spectrogram: Vec<f32>,
    language_token: &mut Option<u32>,
    language: &mut Option<String>,
    english_only: bool,
) -> candle_core::Result<String> {
    model.reset_kv_cache();
    let mel = Tensor::from_vec(
        spectrogram,
        (1, model.config.num_mel_bins, whisper::N_FRAMES),
        device,
    )?;
    let features = model.encoder.forward(&mel, true)?;
    if !english_only && language_token.is_none() {
        let input = Tensor::new(&[[special.sot]], device)?;
        let hidden = model.decoder.forward(&input, &features, true)?;
        let logits = model
            .decoder
            .final_linear(&hidden)?
            .i((0, 0))?
            .to_vec1::<f32>()?;
        let detected = LANGUAGES
            .split_whitespace()
            .filter_map(|code| {
                let id = tokenizer.token_to_id(&format!("<|{code}|>"))?;
                let logit = *logits.get(id as usize)?;
                logit.is_finite().then_some((code, id, logit))
            })
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .ok_or_else(|| {
                candle_core::Error::Msg("Whisper tokenizer has no language tokens".into())
            })?;
        *language_token = Some(detected.1);
        *language = Some(detected.0.into());
        model.decoder.reset_kv_cache();
    }
    let mut prompt = vec![special.sot];
    if let Some(id) = language_token {
        prompt.push(*id);
    }
    prompt.extend([special.transcribe, special.no_timestamps]);
    let prefix = prompt.len();
    let mut no_speech_probability = 0.0;
    let mut log_probability = 0.0;
    let max_tokens = model.config.max_target_positions - prefix;
    for step in 0..max_tokens {
        // Candle 0.11 caches cross-attention K/V only. Self-attention and positions
        // start at zero on every call, so keep the full prompt, reusing the audio
        // cache after the first step (not flushing it for each growing prompt).
        let input = Tensor::new(prompt.as_slice(), device)?.unsqueeze(0)?;
        let hidden = model.decoder.forward(&input, &features, step == 0)?;
        if step == 0 {
            let first = model
                .decoder
                .final_linear(&hidden.narrow(1, 0, 1)?)?
                .i((0, 0))?
                .to_vec1::<f32>()?;
            no_speech_probability = probability(&first, special.no_speech as usize);
        }
        let last = hidden.dim(1)? - 1;
        let mut logits = model
            .decoder
            .final_linear(&hidden.narrow(1, last, 1)?)?
            .i((0, 0))?
            .to_vec1::<f32>()?;
        suppress_logits(
            &mut logits,
            special.eot as usize,
            &model.config.suppress_tokens,
            step == 0,
        );
        let next = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(id, _)| id)
            .ok_or_else(|| {
                candle_core::Error::Msg("Whisper produced no finite text logits".into())
            })?;
        log_probability += probability(&logits, next).max(f64::MIN_POSITIVE).ln();
        if next == special.eot as usize {
            if no_speech_probability > whisper::NO_SPEECH_THRESHOLD
                && log_probability / ((step + 1) as f64) < whisper::LOGPROB_THRESHOLD
            {
                return Ok(String::new());
            }
            return tokenizer
                .decode(&prompt[prefix..], true)
                .map(|s| s.trim().to_owned())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Whisper text decoding failed: {e}"))
                });
        }
        prompt.push(next as u32);
    }
    candle_core::bail!("Whisper reached its text context limit; try a shorter or cleaner recording")
}

fn suppress_logits(logits: &mut [f32], eot: usize, suppressed: &[u32], first: bool) {
    logits[eot + 1..].fill(f32::NEG_INFINITY);
    for &id in suppressed {
        logits[id as usize] = f32::NEG_INFINITY;
    }
    if first {
        logits[eot] = f32::NEG_INFINITY;
        logits[220] = f32::NEG_INFINITY;
    }
}

fn probability(logits: &[f32], index: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator: f64 = logits.iter().map(|v| f64::from(*v - max).exp()).sum();
    f64::from(logits[index] - max).exp() / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable() {
        assert!((probability(&[10000.0, 10000.0], 0) - 0.5).abs() < 1e-8);
        assert_eq!(probability(&[0.0, f32::NEG_INFINITY], 0), 1.0);
    }

    #[test]
    fn suppression_matches_original_mask_for_first_and_later_steps() {
        let eot = 500;
        let suppressed = [0, 42, 220, 503];
        for first in [true, false] {
            let mut logits = vec![1.0; 512];
            suppress_logits(&mut logits, eot, &suppressed, first);
            for (id, actual) in logits.into_iter().enumerate() {
                let blocked = id > eot
                    || suppressed.contains(&(id as u32))
                    || first && (id == eot || id == 220);
                assert_eq!(actual, if blocked { f32::NEG_INFINITY } else { 1.0 });
            }
        }
    }

    fn synthetic_whisper() -> Whisper {
        let config: Config = serde_json::from_value(serde_json::json!({
            "num_mel_bins":80, "max_source_positions":4, "d_model":8,
            "encoder_attention_heads":2, "encoder_layers":1, "vocab_size":16,
            "max_target_positions":16, "decoder_attention_heads":2, "decoder_layers":1
        }))
        .unwrap();
        let variables = candle_nn::VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, whisper::DTYPE, &Device::Cpu);
        // Discover the pinned model's tensor shapes, then supply deterministic,
        // nonzero synthetic parameters, including unit layer-norm scales.
        drop(Whisper::load(&builder, config.clone()).unwrap());
        for (name, var) in variables.data().lock().unwrap().iter() {
            let phase = name.bytes().map(u32::from).sum::<u32>() as f32;
            let values: Vec<f32> = (0..var.elem_count())
                .map(|i| {
                    if var.dims().len() == 1 && name.ends_with(".weight") {
                        1.0
                    } else {
                        ((i as f32 + phase) * 0.37).sin() * 0.2
                    }
                })
                .collect();
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu).unwrap())
                .unwrap();
        }
        Whisper::load(&builder, config).unwrap()
    }

    fn max_difference(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn candle_011_full_prompt_reuses_only_cross_attention_cache() {
        let mut decoder = synthetic_whisper().decoder;
        let mut recomputed = decoder.clone();
        let features = Tensor::from_vec(
            (0..32).map(|i| (i as f32 * 0.19).sin()).collect(),
            (1, 4, 8),
            &Device::Cpu,
        )
        .unwrap();
        let prompt = [1_u32, 2, 3, 4, 5, 6];
        for length in 2..=prompt.len() {
            let input = Tensor::new(&prompt[..length], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let cached = decoder.forward(&input, &features, length == 2).unwrap();
            let fresh = recomputed.forward(&input, &features, true).unwrap();
            assert!(max_difference(&cached, &fresh) < 1e-5);
        }
        let input = Tensor::new(&prompt[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full = decoder
            .forward(&input, &features, false)
            .unwrap()
            .narrow(1, 5, 1)
            .unwrap();
        let last_only = Tensor::new(&[[6_u32]], &Device::Cpu).unwrap();
        let wrong = decoder.forward(&last_only, &features, false).unwrap();
        assert!(max_difference(&full, &wrong) > 1e-4);

        let changed_features = features.neg().unwrap();
        let stale = decoder.forward(&input, &changed_features, false).unwrap();
        let flushed = decoder.forward(&input, &changed_features, true).unwrap();
        recomputed.reset_kv_cache();
        let reset = recomputed
            .forward(&input, &changed_features, false)
            .unwrap();
        assert!(max_difference(&flushed, &reset) < 1e-5);
        assert!(max_difference(&stale, &reset) > 1e-4);
    }

    #[test]
    fn rejects_invalid_config_without_allocating_tensors() {
        let mut config: Config = serde_json::from_value(serde_json::json!({
            "num_mel_bins":80, "max_source_positions":1500, "d_model":384,
            "encoder_attention_heads":6, "encoder_layers":4, "vocab_size":51865,
            "max_target_positions":448, "decoder_attention_heads":6, "decoder_layers":4
        }))
        .unwrap();
        assert!(validate_config(&config).is_ok());
        config.encoder_attention_heads = 0;
        assert!(validate_config(&config).is_err());
        config.encoder_attention_heads = 6;
        config.d_model = usize::MAX;
        assert!(validate_config(&config).is_err());
        config.encoder_attention_heads = 1;
        config.decoder_attention_heads = 1;
        // Candle's encoder sinusoids divide by (d_model / 2 - 1).
        config.d_model = 2;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn rejects_wrong_model_format_and_missing_assets() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(
            transcribe(file.path(), &[0.1; 10], None, 1, false)
                .unwrap_err()
                .contains("GGML/GGUF")
        );
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"not json").unwrap();
        assert!(
            transcribe(dir.path(), &[0.1; 10], None, 1, false)
                .unwrap_err()
                .contains("config.json")
        );
    }
}
