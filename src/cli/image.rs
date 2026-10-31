use std::path::PathBuf;

use super::{
    Action,
    args::{Args, Flag, number},
    help,
};
use crate::config::{ImageConfig, Job, Plan};

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut model = None;
    let mut prompt = None;
    let mut positional = Vec::new();
    let mut dry_run = false;
    let mut config = ImageConfig {
        model_dir: PathBuf::new(),
        tokenizer: None,
        prompt: String::new(),
        negative_prompt: String::new(),
        output: PathBuf::from("image.png"),
        width: 512,
        height: 512,
        steps: 20,
        seed: 42,
        guidance: 7.5,
        device: "auto".into(),
        attention_slicing: None,
    };
    while let Some(argument) = args.next() {
        let Some(flag) = Flag::parse(&argument) else {
            positional.push(argument);
            continue;
        };
        match flag.name.as_str() {
            "--help" | "--help-all" | "-h" => {
                flag.switch()?;
                return Ok(Action::Help(help::IMAGE));
            }
            "--model" | "--model-dir" | "-m" => {
                if model.replace(PathBuf::from(args.value(&flag)?)).is_some() {
                    return Err("model directory specified twice".into());
                }
            }
            "--prompt" | "-p" => {
                if prompt.replace(args.text(&flag)?).is_some() {
                    return Err("prompt specified twice".into());
                }
            }
            "--negative-prompt" | "-n" => config.negative_prompt = args.text(&flag)?,
            "--output" | "--out" | "-o" => config.output = PathBuf::from(args.value(&flag)?),
            "--tokenizer" => config.tokenizer = Some(PathBuf::from(args.value(&flag)?)),
            "--width" => {
                config.width = u32::try_from(number(&args.text(&flag)?, "width", false)?)
                    .map_err(|_| "width is too large")?
            }
            "--height" => {
                config.height = u32::try_from(number(&args.text(&flag)?, "height", false)?)
                    .map_err(|_| "height is too large")?
            }
            "--steps" => {
                config.steps = usize::try_from(number(&args.text(&flag)?, "steps", false)?)
                    .map_err(|_| "steps is too large")?
            }
            "--seed" => config.seed = number(&args.text(&flag)?, "seed", true)?,
            "--guidance" => {
                config.guidance = args
                    .text(&flag)?
                    .parse::<f64>()
                    .map_err(|_| "guidance must be a number")?
            }
            "--device" => config.device = args.text(&flag)?,
            "--attention-slicing" => {
                config.attention_slicing = Some(
                    usize::try_from(number(&args.text(&flag)?, "attention slicing", false)?)
                        .map_err(|_| "attention slicing is too large")?,
                )
            }
            "--dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            _ => {
                return Err(format!(
                    "unknown image option '{}'; run `lma h i`",
                    flag.name
                ));
            }
        }
    }
    let mut positional = positional.into_iter();
    config.model_dir = model
        .or_else(|| positional.next().map(PathBuf::from))
        .ok_or("image needs a local SD1.5 model directory; use `lma i -m DIR -p TEXT` or `lma nss image --model-dir DIR --prompt TEXT`")?;
    config.prompt = prompt
        .or_else(|| positional.next().and_then(|s| s.into_string().ok()))
        .ok_or("image needs a UTF-8 prompt; use -p TEXT at root or --prompt TEXT in NSS")?;
    if positional.next().is_some() {
        return Err("too many image arguments; quote the prompt".into());
    }
    if config.model_dir.as_os_str().is_empty() || config.prompt.trim().is_empty() {
        return Err("model directory and prompt must not be empty".into());
    }
    if !(64..=1024).contains(&config.width)
        || !(64..=1024).contains(&config.height)
        || !config.width.is_multiple_of(8)
        || !config.height.is_multiple_of(8)
    {
        return Err("width and height must be 64..1024 and divisible by 8".into());
    }
    if !(1..=150).contains(&config.steps) {
        return Err("steps must be between 1 and 150".into());
    }
    if !config.guidance.is_finite() || !(1.0..=20.0).contains(&config.guidance) {
        return Err("guidance must be finite and between 1 and 20".into());
    }
    if !matches!(config.device.as_str(), "auto" | "cpu" | "metal" | "cuda") {
        return Err("device must be auto, cpu, metal, or cuda".into());
    }
    if let Some(size) = config.attention_slicing {
        let heads = if config.guidance > 1.0 { 16 } else { 8 };
        if size == 0 || size > heads || heads % size != 0 {
            return Err(format!(
                "attention slicing must be a positive divisor of {heads}"
            ));
        }
    }
    Ok(Action::Run(Box::new(Plan {
        job: Job::Image(config),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    #[test]
    fn constructs_a_local_image_plan() {
        let plan = plan(&[
            "nss",
            "image-gen",
            "models/sd15",
            "a red fox",
            "--steps",
            "10",
            "--seed",
            "7",
            "--dry-run",
        ]);
        assert!(plan.dry_run);
        let Job::Image(config) = plan.job else {
            panic!()
        };
        assert_eq!(config.prompt, "a red fox");
        assert_eq!(config.steps, 10);
        assert_eq!(config.width, 512);
        assert_eq!(config.seed, 7);
    }

    #[test]
    fn canonical_image_flags_and_nss_aliases_preserve_configuration() {
        let root = plan(&[
            "i",
            "-m",
            "sd15",
            "-p",
            "a fox",
            "-n",
            "blurry",
            "-o",
            "fox.png",
            "-t",
            "tokenizer.json",
            "-w",
            "256",
            "-ht",
            "128",
            "-s",
            "10",
            "-sd",
            "7",
            "-g",
            "1",
            "-de",
            "cpu",
            "-a",
            "2",
            "-d",
        ]);
        assert!(root.dry_run);
        let Job::Image(root) = root.job else { panic!() };
        assert_eq!(root.model_dir, PathBuf::from("sd15"));
        assert_eq!(root.prompt, "a fox");
        assert_eq!(root.negative_prompt, "blurry");
        assert_eq!(root.output, PathBuf::from("fox.png"));
        assert_eq!(root.tokenizer, Some(PathBuf::from("tokenizer.json")));
        assert_eq!(
            (root.width, root.height, root.steps, root.seed),
            (256, 128, 10, 7)
        );
        assert_eq!(root.guidance, 1.0);
        assert_eq!(root.device, "cpu");
        assert_eq!(root.attention_slicing, Some(2));
        for command in ["image", "image-gen", "imagen", "img"] {
            let parsed = plan(&[
                "nss",
                command,
                "--model-dir",
                "sd15",
                "-p",
                "a fox",
                "-n",
                "blurry",
                "--output",
                "fox.png",
                "--tokenizer",
                "tokenizer.json",
                "--width",
                "256",
                "--height",
                "128",
                "--steps",
                "10",
                "--seed",
                "7",
                "--guidance",
                "1",
                "--device",
                "cpu",
                "--attention-slicing",
                "2",
                "--dry-run",
            ]);
            assert!(parsed.dry_run);
            let Job::Image(config) = parsed.job else {
                panic!()
            };
            assert_eq!(
                serde_json::to_value(config).unwrap(),
                serde_json::to_value(&root).unwrap()
            );
        }
    }

    #[test]
    fn rejects_invalid_sizes_and_sampling() {
        for flags in [
            vec!["--width", "511"],
            vec!["--steps", "151"],
            vec!["--guidance", "NaN"],
            vec!["--device", "unknown"],
            vec!["--attention-slicing", "3"],
        ] {
            let argv = [vec!["nss", "image", "sd15", "prompt"], flags].concat();
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
