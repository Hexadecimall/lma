use std::{collections::HashSet, path::PathBuf};

use super::{
    Action,
    args::{Args, Flag},
};
use crate::{
    config::{Job, Plan},
    speech::{self, TranscribeConfig},
};

pub const HELP: &str = "lma t — local, in-process Whisper speech recognition

USAGE
  lma t -m DIR [OPTIONS] AUDIO.wav
  lma t -m DIR -a AUDIO.wav -l en -t 2 -g off -j -d
  lma nss transcribe --model DIR --audio AUDIO.wav --language en --threads 2 --gpu off --json --dry-run

ROOT OPTIONS (canonical short flags only)
  -m DIR          Local Hugging Face Whisper model directory (required)
  -a WAV          Input WAV, alternatively one positional WAV path
  -l CODE         Whisper language code (en, fr, zh, ...), or auto (default)
  -t N            Spectral preprocessing workers, 1..256 (default: up to 8)
  -g on|off       GPU only when compiled for Metal/macOS or CUDA; otherwise CPU
  -j              Print text, language, and segments as JSON
  -d              Print the plan without opening audio or loading any model
  -h              Show this help

NSS OPTIONS
  lma nss transcribe accepts --model/-m, --audio/-a, --language, --threads/-t,
  --gpu, --json, --dry-run, and --help/-h (also --help-all).
  Only NSS accepts -- as an option terminator: lma nss transcribe -m DIR -- -audio.wav
  Long flags and bare -- are not supported at root; use lma h t or lma nss help transcribe.

MODEL
  DIR must contain config.json, tokenizer.json, and model.safetensors from a
  compatible Hugging Face Whisper model, e.g. openai/whisper-tiny or tiny.en.
  Only local, single-file safetensors weights up to 4 GiB are accepted.
  whisper.cpp GGML/GGUF models are not compatible. Nothing is downloaded.

AUDIO AND OUTPUT
  PCM16 or float32 WAV, mono/stereo, 8–192 kHz; native band-limited resampling
  to mono 16 kHz. Maximum 30 minutes and 512 MiB. Default output is plain text.
  JSON segment times are chunk boundaries (up to 30 seconds), not word alignment.
  -g on (NSS: --gpu on) fails if unavailable; there is no silent CPU fallback.
  Threads controls preprocessing; Candle inference uses its process-wide pool
  (RAYON_NUM_THREADS). Decoding is deterministic greedy transcription, not translation.
  JSON stays machine-readable without terminal styling; redirected text stays plain.
  Model and audio paths remain yours; lma does not move them into ~/.lma.
";

pub(super) fn parse(args: Args) -> Result<Action, String> {
    match parse_options(args)? {
        None => Ok(Action::Help(HELP)),
        Some((config, dry_run)) => Ok(Action::Run(Box::new(Plan {
            job: Job::Transcribe(config),
            dry_run,
        }))),
    }
}

fn parse_options(mut args: Args) -> Result<Option<(TranscribeConfig, bool)>, String> {
    let mut config = TranscribeConfig {
        model: PathBuf::new(),
        audio: PathBuf::new(),
        language: None,
        threads: speech::default_threads(),
        gpu: speech::gpu_available(),
        json: false,
    };
    let mut dry_run = false;
    let mut seen = HashSet::new();
    let mut positional = Vec::new();
    while let Some(argument) = args.next() {
        if argument == "--" {
            positional.extend(args.rest());
            break;
        }
        let Some(flag) = Flag::parse(&argument) else {
            positional.push(argument);
            continue;
        };
        let canonical = match flag.name.as_str() {
            "--help" | "--help-all" | "-h" => {
                flag.switch()?;
                return Ok(None);
            }
            "--model" | "-m" => "model",
            "--audio" | "-a" => "audio",
            "--language" => "language",
            "--threads" | "-t" => "threads",
            "--gpu" => "gpu",
            "--json" => "json",
            "--dry-run" => "dry-run",
            _ => {
                return Err(format!(
                    "unknown transcribe option '{}'; run `lma h t`",
                    flag.name
                ));
            }
        };
        if !seen.insert(canonical) {
            return Err(format!("transcribe {canonical} specified twice"));
        }
        match canonical {
            "model" => config.model = PathBuf::from(args.value(&flag)?),
            "audio" => config.audio = PathBuf::from(args.value(&flag)?),
            "language" => {
                let code = args.text(&flag)?;
                config.language = (code != "auto").then_some(code);
            }
            "threads" => {
                config.threads = args
                    .text(&flag)?
                    .parse::<usize>()
                    .map_err(|_| "speech threads must be an integer between 1 and 256")?
            }
            "gpu" => {
                config.gpu = match args.text(&flag)?.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => {
                        return Err(
                            "transcribe GPU requires on or off (-g at root, --gpu in NSS)".into(),
                        );
                    }
                }
            }
            "json" => {
                flag.switch()?;
                config.json = true;
            }
            "dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            _ => unreachable!(),
        }
    }
    if positional.len() > 1 || !positional.is_empty() && seen.contains("audio") {
        return Err(
            "transcribe needs exactly one audio path, positional or -a/--audio (not both)".into(),
        );
    }
    if let Some(audio) = positional.pop() {
        config.audio = PathBuf::from(audio);
    }
    if config.model.as_os_str().is_empty() {
        return Err("transcribe requires -m DIR (NSS: --model DIR) containing local Whisper safetensors, config.json, and tokenizer.json".into());
    }
    if config.audio.as_os_str().is_empty() {
        return Err(
            "transcribe requires a WAV audio path; use -a FILE.wav (NSS: --audio FILE.wav) or a positional path"
                .into(),
        );
    }
    speech::validate_options(&config.options())?;
    Ok(Some((config, dry_run)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn nss_options(argv: &[&str]) -> Result<Option<(TranscribeConfig, bool)>, String> {
        parse_options(Args::new(argv.iter().map(OsString::from)))
    }

    #[test]
    fn dry_run_does_not_require_assets() {
        let (config, dry_run) = nss_options(&[
            "-m",
            "not-installed",
            "not-recorded.wav",
            "--dry-run",
            "--json",
            "--gpu=off",
            "--language=en",
            "-t",
            "2",
        ])
        .unwrap()
        .unwrap();
        assert!(dry_run && config.json);
        assert!(!config.gpu);
        assert_eq!(config.threads, 2);
        assert_eq!(config.language.as_deref(), Some("en"));
        assert_eq!(config.audio, PathBuf::from("not-recorded.wav"));
    }

    #[test]
    fn audio_flags_auto_language_and_terminator() {
        let (config, _) = nss_options(&["--model=dir", "-a", "test.wav", "--language", "auto"])
            .unwrap()
            .unwrap();
        assert!(config.language.is_none());
        let (config, _) = nss_options(&["-m", "dir", "--", "-audio.wav"])
            .unwrap()
            .unwrap();
        assert_eq!(config.audio, PathBuf::from("-audio.wav"));
        assert!(nss_options(&["--help"]).unwrap().is_none());
    }

    #[test]
    fn rejects_ambiguous_or_invalid_arguments() {
        for argv in [
            vec![],
            vec!["test.wav"],
            vec!["-m", "dir"],
            vec!["-m", "dir", "a.wav", "b.wav"],
            vec!["-m", "dir", "a.wav", "--audio", "b.wav"],
            vec!["-m", "dir", "a.wav", "--threads", "0"],
            vec!["-m", "dir", "a.wav", "--threads", "257"],
            vec!["-m", "dir", "a.wav", "--threads", "auto"],
            vec!["-m", "dir", "a.wav", "--gpu"],
            vec!["-m", "dir", "a.wav", "--gpu", "auto"],
            vec!["-m", "dir", "a.wav", "--json=yes"],
            vec!["-m", "dir", "a.wav", "--language", "english"],
            vec!["-m", "dir", "a.wav", "-a", "a.wav"],
            vec!["-m", "dir", "a.wav", "--unknown"],
            vec!["-m", "dir", "a.wav", "--model", "other"],
        ] {
            assert!(nss_options(&argv).is_err(), "accepted {argv:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn paths_need_not_be_utf8() {
        use std::os::unix::ffi::OsStringExt;
        let audio = OsString::from_vec(vec![0xff, b'.', b'w', b'a', b'v']);
        let (config, _) = parse_options(Args::new(["-m".into(), "dir".into(), audio.clone()]))
            .unwrap()
            .unwrap();
        assert_eq!(config.audio.as_os_str(), audio);
    }

    #[test]
    fn canonical_short_flags_and_nss_aliases_build_the_same_transcription() {
        for argv in [
            vec![
                "t",
                "-m",
                "dir",
                "-a",
                "audio.wav",
                "-l",
                "en",
                "-t",
                "2",
                "-g",
                "off",
                "-j",
                "-d",
            ],
            vec![
                "nss",
                "transcribe",
                "--model",
                "dir",
                "--audio",
                "audio.wav",
                "--language",
                "en",
                "--threads",
                "2",
                "--gpu",
                "off",
                "--json",
                "--dry-run",
            ],
            vec![
                "nss",
                "transcribe",
                "-m",
                "dir",
                "-a",
                "audio.wav",
                "--language=en",
                "-t",
                "2",
                "--gpu=off",
                "--json",
                "--dry-run",
            ],
        ] {
            let parsed = crate::cli::tests::plan(&argv);
            assert!(parsed.dry_run);
            let Job::Transcribe(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.model, PathBuf::from("dir"));
            assert_eq!(config.audio, PathBuf::from("audio.wav"));
            assert_eq!(config.language.as_deref(), Some("en"));
            assert_eq!(config.threads, 2);
            assert!(!config.gpu);
            assert!(config.json);
        }
    }

    #[test]
    fn only_nss_accepts_the_audio_option_terminator() {
        let parsed = crate::cli::tests::plan(&[
            "nss",
            "transcribe",
            "-m",
            "dir",
            "--gpu=off",
            "--",
            "-audio.wav",
        ]);
        let Job::Transcribe(config) = parsed.job else {
            panic!()
        };
        assert_eq!(config.audio, PathBuf::from("-audio.wav"));
        let Err(error) =
            crate::cli::parse(["t", "-m", "dir", "--", "-audio.wav"].map(OsString::from))
        else {
            panic!("root accepted the NSS terminator")
        };
        assert!(error.contains("unsupported root option '--'"), "{error}");
        for argv in [vec!["t", "-h"], vec!["nss", "transcribe", "--help"]] {
            let Action::Help(help) =
                crate::cli::parse(argv.into_iter().map(OsString::from)).unwrap()
            else {
                panic!("expected transcription help")
            };
            assert_eq!(help, HELP);
            assert!(help.contains("lma t -m DIR -a AUDIO.wav -l en -t 2 -g off -j -d"));
            assert!(help.contains("lma nss transcribe --model DIR --audio AUDIO.wav"));
        }
    }

    #[test]
    fn wraps_the_parent_job_contract() {
        let action = parse(Args::new(
            ["-m", "dir", "audio.wav"].into_iter().map(OsString::from),
        ))
        .unwrap();
        let Action::Run(plan) = action else {
            panic!("expected run plan")
        };
        assert!(matches!(plan.job, Job::Transcribe(_)));
    }
}
