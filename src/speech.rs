//! Local, in-process Whisper transcription, independent of the llama worker.
//!
//! `model` is a Hugging Face/Candle Whisper directory containing `config.json`,
//! `tokenizer.json`, and a single `model.safetensors` (not whisper.cpp GGML/GGUF).
//! Inference never downloads models; model installation is separate from speech.
//! `threads` limits spectral preprocessing workers; Candle
//! 0.11 controls inference threads through its process-wide pool/RAYON_NUM_THREADS.
//! Segments are audio chunk boundaries (up to 30 seconds), not word alignment.

use serde::Serialize;
use std::path::{Path, PathBuf};

/// Check the directory/asset layout without loading weights or downloading anything.
pub fn validate_model_directory(directory: &Path) -> Result<(), String> {
    if !directory.is_dir() {
        return Err("Choose a local Hugging Face Whisper directory containing config.json, tokenizer.json, and model.safetensors (not a GGML/GGUF file).".into());
    }
    for (name, limit) in [
        ("config.json", 1024 * 1024),
        ("tokenizer.json", 32 * 1024 * 1024),
        ("model.safetensors", 4u64 * 1024 * 1024 * 1024),
    ] {
        let file = crate::local_file::open_regular(&directory.join(name))?;
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        if length == 0 || length > limit {
            return Err(format!(
                "Whisper {name} must be nonempty and at most {limit} bytes"
            ));
        }
    }
    Ok(())
}

pub const SAMPLE_RATE: usize = 16_000;
pub const MAX_DICTATION_SECONDS: usize = 120;
pub const MAX_AUDIO_SECONDS: usize = 30 * 60;
pub const MAX_THREADS: usize = 256;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionOptions {
    pub model: PathBuf,
    pub audio: PathBuf,
    /// Whisper language code, or None for automatic detection.
    pub language: Option<String>,
    pub threads: usize,
    /// Request a compiled GPU backend; failure never silently falls back to CPU.
    pub gpu: bool,
}

/// CLI job payload; kept separate from the shared inference options.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeConfig {
    pub model: PathBuf,
    pub audio: PathBuf,
    pub language: Option<String>,
    pub threads: usize,
    pub gpu: bool,
    pub json: bool,
}

impl TranscribeConfig {
    pub fn options(&self) -> TranscriptionOptions {
        TranscriptionOptions {
            model: self.model.clone(),
            audio: self.audio.clone(),
            language: self.language.clone(),
            threads: self.threads,
            gpu: self.gpu,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(1)
}

pub const fn gpu_available() -> bool {
    cfg!(all(
        feature = "speech",
        any(
            all(feature = "metal", target_os = "macos"),
            all(feature = "cuda", not(target_os = "macos"))
        )
    ))
}

/// Validate syntax only: safe for `--dry-run` without local assets.
pub fn validate_options(options: &TranscriptionOptions) -> Result<(), String> {
    if options.model.as_os_str().is_empty() || options.audio.as_os_str().is_empty() {
        return Err("transcribe requires non-empty model and audio paths".into());
    }
    validate_settings(options.language.as_deref(), options.threads, options.gpu)
}

fn validate_settings(language: Option<&str>, threads: usize, gpu: bool) -> Result<(), String> {
    if !(1..=MAX_THREADS).contains(&threads) {
        return Err(format!(
            "speech threads must be between 1 and {MAX_THREADS}"
        ));
    }
    if let Some(language) = language
        && !LANGUAGES.split_whitespace().any(|code| code == language)
    {
        return Err(format!(
            "unsupported Whisper language code {language:?}; use a code such as en, fr, or zh, or auto"
        ));
    }
    if gpu && !gpu_available() {
        return Err("speech GPU requested but no supported GPU backend is compiled; build with --features metal (macOS) or cuda, or use -g off".into());
    }
    Ok(())
}

const LANGUAGES: &str = "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue";

/// Transcribe a bounded local PCM16/float32 WAV (mono or stereo, <=30 minutes).
pub fn transcribe(options: TranscriptionOptions) -> Result<Transcript, String> {
    validate_options(&options)?;
    #[cfg(feature = "speech")]
    {
        let samples = audio::read_wav(&options.audio)?;
        backend::transcribe(
            &options.model,
            &samples,
            options.language.as_deref(),
            options.threads,
            options.gpu,
        )
    }
    #[cfg(not(feature = "speech"))]
    Err("speech support is disabled; rebuild with --features speech".into())
}

/// Transcribe mono, 16 kHz floats in [-1, 1], already captured by the caller.
/// Rejects empty/non-finite/out-of-range input and recordings over 120 seconds.
/// This is blocking: desktop callers should use a dedicated blocking task.
pub fn transcribe_samples(
    model: &Path,
    samples: &[f32],
    language: Option<&str>,
    threads: usize,
    gpu: bool,
) -> Result<Transcript, String> {
    validate_settings(language, threads, gpu)?;
    validate_samples(samples, MAX_DICTATION_SECONDS)?;
    #[cfg(feature = "speech")]
    {
        backend::transcribe(model, samples, language, threads, gpu)
    }
    #[cfg(not(feature = "speech"))]
    {
        let _ = model;
        Err("speech support is disabled; rebuild with --features speech".into())
    }
}

fn validate_samples(samples: &[f32], seconds: usize) -> Result<(), String> {
    if samples.is_empty() {
        return Err("audio contains no samples".into());
    }
    if samples.len() > SAMPLE_RATE * seconds {
        return Err(format!("audio exceeds the {seconds}-second limit"));
    }
    if samples.iter().any(|s| !s.is_finite() || s.abs() > 1.0) {
        return Err("audio samples must be finite floats in [-1, 1]".into());
    }
    Ok(())
}

#[cfg(feature = "speech")]
#[path = "speech/audio.rs"]
mod audio;
#[cfg(feature = "speech")]
#[path = "speech/backend.rs"]
mod backend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_samples_before_model_loading() {
        for samples in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![1.01]] {
            assert!(transcribe_samples(Path::new("missing"), &samples, None, 1, false).is_err());
        }
        assert!(validate_samples(&vec![0.0; SAMPLE_RATE * 120 + 1], 120).is_err());
        assert!(validate_samples(&[-1.0, 0.0, 1.0], 120).is_ok());
    }

    #[test]
    fn validates_settings_without_assets() {
        let mut options = TranscriptionOptions {
            model: "not-installed".into(),
            audio: "not-recorded.wav".into(),
            language: Some("en".into()),
            threads: 1,
            gpu: false,
        };
        assert!(validate_options(&options).is_ok());
        options.threads = 0;
        assert!(validate_options(&options).is_err());
        options.threads = MAX_THREADS + 1;
        assert!(validate_options(&options).is_err());
        options.threads = 1;
        options.language = Some("english".into());
        assert!(validate_options(&options).is_err());
        if !gpu_available() {
            assert!(validate_settings(None, 1, true).is_err());
        }
    }

    #[test]
    fn model_layout_requires_all_regular_nonempty_assets() {
        let directory = tempfile::tempdir().unwrap();
        assert!(validate_model_directory(directory.path()).is_err());
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(directory.path().join(name), b"fixture").unwrap();
        }
        assert!(validate_model_directory(directory.path()).is_ok());
        let weights = directory.path().join("model.safetensors");
        std::fs::write(&weights, b"").unwrap();
        assert!(validate_model_directory(directory.path()).is_err());
        std::fs::remove_file(&weights).unwrap();
        std::fs::create_dir(&weights).unwrap();
        assert!(validate_model_directory(directory.path()).is_err());
    }

    #[test]
    fn transcript_json_contract() {
        let transcript = Transcript {
            text: "hello".into(),
            language: Some("en".into()),
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 100,
                text: "hello".into(),
            }],
        };
        let json = serde_json::to_value(transcript).unwrap();
        assert_eq!(json["segments"][0]["end_ms"], 100);
        assert_eq!(json["language"], "en");
    }

    #[test]
    fn links_llama_alongside_speech_without_initializing_it() {
        // Retain the native llama references in this component's test executable.
        let _ = std::hint::black_box(llama_cpp_2::llama_backend::LlamaBackend::init);
    }

    #[cfg(feature = "speech")]
    #[test]
    fn missing_model_is_actionable() {
        let dir = tempfile::tempdir().unwrap();
        let error = transcribe_samples(dir.path(), &[0.0; 100], None, 1, false).unwrap_err();
        assert!(error.contains("config.json"), "{error}");
    }

    #[cfg(not(feature = "speech"))]
    #[test]
    fn disabled_backend_is_actionable() {
        assert!(
            transcribe_samples(Path::new("model"), &[0.0], None, 1, false)
                .unwrap_err()
                .contains("--features speech")
        );
    }
}
