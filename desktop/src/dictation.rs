//! Local dictation: persist only a model path, never microphone samples.
//! Model installation is a separate, explicit opt-in command.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use tauri::Manager;
use tokio::sync::Mutex as AsyncMutex;

const MAX_SAMPLES: usize = 16_000 * 120;
const MAX_ENCODED: usize = (MAX_SAMPLES * 2).div_ceil(3) * 4;
const MAX_SETTINGS: u64 = 16 * 1024;
const FILE_NAME: &str = "dictation.json";
static STORAGE: Mutex<()> = Mutex::new(());
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
pub struct DictationState(Arc<AsyncMutex<()>>);

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Settings {
    pub model_path: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    version: u32,
    settings: Settings,
}

fn validate_path(path: &str) -> Result<(), String> {
    if path.is_empty() || path.len() > 4096 || path.contains('\0') || !Path::new(path).is_absolute()
    {
        return Err(
            "Enter an absolute path to a local Whisper model directory (maximum 4096 bytes)."
                .into(),
        );
    }
    Ok(())
}

fn load(directory: &Path) -> Result<Settings, String> {
    let path = lma::models::storage::read_path(directory, FILE_NAME)?;
    let before = match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Settings::default()),
        Err(e) => return Err(format!("Cannot inspect dictation settings: {e}")),
        Ok(meta) => meta,
    };
    if !before.is_file() || before.file_type().is_symlink() || before.len() > MAX_SETTINGS {
        return Err("dictation.json must be a regular, non-symlink file of at most 16 KiB.".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800); // O_NOFOLLOW | O_NONBLOCK
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options
        .open(path)
        .map_err(|e| format!("Cannot read dictation settings: {e}"))?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > MAX_SETTINGS {
        return Err("Invalid dictation settings file.".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != metadata.dev() || before.ino() != metadata.ino() {
            return Err("Dictation settings changed while opening them.".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(MAX_SETTINGS + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_SETTINGS {
        return Err("dictation.json exceeds 16 KiB.".into());
    }
    let saved: Saved = serde_json::from_slice(&bytes)
        .map_err(|_| "Invalid dictation.json. Repair or remove it to choose a model.")?;
    if saved.version != 1 {
        return Err("Unsupported dictation.json version.".into());
    }
    if let Some(path) = &saved.settings.model_path {
        validate_path(path)?;
    }
    Ok(saved.settings)
}

fn model_path(input: &str) -> Result<PathBuf, String> {
    validate_path(input)?;
    let path = fs::canonicalize(input).map_err(|e| {
        format!("Cannot open Whisper model; check the path and file permissions: {e}")
    })?;
    lma::speech::validate_model_directory(&path)?;
    whisper_config(&path)?;
    Ok(path)
}

fn save(directory: &Path, input: &str) -> Result<Settings, String> {
    let path = model_path(input)?;
    let model_path = path
        .to_str()
        .ok_or("Whisper model path must be UTF-8.")?
        .to_owned();
    validate_path(&model_path)?;
    load(directory)?; // Never overwrite unknown, damaged or redirected settings.
    let settings = Settings {
        model_path: Some(model_path),
    };
    let bytes = serde_json::to_vec(&Saved {
        version: 1,
        settings: settings.clone(),
    })
    .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_SETTINGS {
        return Err("Dictation settings exceed 16 KiB.".into());
    }
    fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let temporary = directory.join(format!(
        ".dictation-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).map_err(|e| e.to_string())?;
    let result = (|| {
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::rename(&temporary, directory.join(FILE_NAME)).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(settings)
}

fn decode_pcm(encoded: &str) -> Result<Vec<f32>, String> {
    if encoded.is_empty() || encoded.len() > MAX_ENCODED {
        return Err(
            "Dictation audio must be nonempty and at most 120 seconds of mono 16 kHz PCM16.".into(),
        );
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| "Invalid PCM16 base64 audio.")?;
    if bytes.is_empty() || bytes.len() % 2 != 0 || bytes.len() > MAX_SAMPLES * 2 {
        return Err("Invalid PCM16 audio length (maximum 120 seconds).".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
        .collect())
}

const MAX_MODEL_JSON: u64 = 1024 * 1024;
const MAX_MODEL_ROOTS: usize = 128;
const MAX_MODEL_ENTRIES: usize = 10_000;
const MAX_MODEL_DEPTH: usize = 8;
const MAX_INSTALLED_MODELS: usize = 512;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    name: String,
    model_path: String,
}

#[derive(Default, Serialize)]
pub struct InstalledModels {
    models: Vec<InstalledModel>,
    warnings: Vec<String>,
}

fn read_model_json(path: &Path) -> Result<serde_json::Value, String> {
    let before = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !before.is_file() || before.file_type().is_symlink() || before.len() > MAX_MODEL_JSON {
        return Err("Model metadata must be a regular, non-symlink file of at most 1 MiB.".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    let file = options.open(path).map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > MAX_MODEL_JSON {
        return Err("Invalid model metadata file.".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != metadata.dev() || before.ino() != metadata.ino() {
            return Err("Model metadata changed while opening it.".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(MAX_MODEL_JSON + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_MODEL_JSON {
        return Err("Model metadata exceeds 1 MiB.".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("Invalid model metadata: {e}"))
}

fn whisper_config(directory: &Path) -> Result<serde_json::Value, String> {
    let config = read_model_json(&directory.join("config.json"))?;
    if config["model_type"].as_str() != Some("whisper") {
        return Err("This dictation backend currently supports model_type: whisper only. Choose a compatible local model directory.".into());
    }
    Ok(config)
}

fn installed_model(directory: &Path) -> Option<InstalledModel> {
    let config = whisper_config(directory).ok()?;
    lma::speech::validate_model_directory(directory).ok()?;
    let source = read_model_json(&directory.join(".lma-source.json")).ok();
    let label = source
        .as_ref()
        .and_then(|value| value["repository"].as_str())
        .or_else(|| config["_name_or_path"].as_str())
        .or_else(|| config["name_or_path"].as_str())
        .or_else(|| config["name"].as_str())
        .filter(|name| !name.trim().is_empty() && name.len() <= 512);
    let path = directory.to_str()?;
    validate_path(path).ok()?;
    Some(InstalledModel {
        name: label.unwrap_or(path).to_owned(),
        model_path: path.to_owned(),
    })
}

fn warning(catalog: &mut InstalledModels, message: String) {
    if catalog.warnings.len() < 16 && !catalog.warnings.contains(&message) {
        catalog.warnings.push(message);
    }
}

fn scan_directory(
    directory: &Path,
    depth: usize,
    remaining: &mut usize,
    visited: &mut HashSet<PathBuf>,
    catalog: &mut InstalledModels,
) {
    if *remaining == 0
        || catalog.models.len() >= MAX_INSTALLED_MODELS
        || !visited.insert(directory.to_owned())
    {
        return;
    }
    *remaining -= 1;
    if let Some(model) = installed_model(directory) {
        catalog.models.push(model);
    }
    if depth >= MAX_MODEL_DEPTH {
        warning(
            catalog,
            "Speech model scan depth limit reached; deeper folders were not searched.".into(),
        );
        return;
    }
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            warning(
                catalog,
                format!("Cannot scan {}: {error}", directory.display()),
            );
            return;
        }
    };
    for entry in entries {
        if *remaining == 0 || catalog.models.len() >= MAX_INSTALLED_MODELS {
            break;
        }
        *remaining -= 1;
        let Ok(entry) = entry else { continue };
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        // Never follow directory symlinks encountered during traversal. Only
        // explicit roots are canonicalized; weights are inspected, not loaded.
        if entry
            .file_type()
            .is_ok_and(|kind| kind.is_dir() && !kind.is_symlink())
        {
            scan_directory(&entry.path(), depth + 1, remaining, visited, catalog);
        }
    }
}

fn scan_models(roots: &[PathBuf], mut remaining: usize) -> InstalledModels {
    let mut catalog = InstalledModels::default();
    let mut visited = HashSet::new();
    if roots.len() > MAX_MODEL_ROOTS {
        warning(&mut catalog, "Speech model scan root limit reached.".into());
    }
    for root in roots.iter().take(MAX_MODEL_ROOTS) {
        let directory = match fs::canonicalize(root) {
            Ok(directory) if directory.is_dir() => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Ok(_) => continue,
            Err(error) => {
                warning(
                    &mut catalog,
                    format!("Cannot inspect {}: {error}", root.display()),
                );
                continue;
            }
        };
        scan_directory(&directory, 0, &mut remaining, &mut visited, &mut catalog);
        if remaining == 0 || catalog.models.len() >= MAX_INSTALLED_MODELS {
            warning(
                &mut catalog,
                "Speech model scan limit reached; some folders were not searched.".into(),
            );
            break;
        }
    }
    catalog
        .models
        .sort_by(|a, b| a.name.cmp(&b.name).then(a.model_path.cmp(&b.model_path)));
    catalog
}

#[tauri::command]
pub async fn dictation_models(app: tauri::AppHandle) -> Result<InstalledModels, String> {
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut roots = lma::models::discovery::default_local_dirs();
        let mut settings_warning = None;
        {
            let _storage = STORAGE
                .lock()
                .map_err(|_| "Dictation settings lock failed.")?;
            match load(&directory) {
                Ok(settings) => {
                    if let Some(path) = settings.model_path {
                        roots.insert(0, path.into());
                    }
                }
                Err(error) => settings_warning = Some(error),
            }
        }
        let mut catalog = scan_models(&roots, MAX_MODEL_ENTRIES);
        if let Some(error) = settings_warning {
            warning(&mut catalog, error);
        }
        Ok(catalog)
    })
    .await
    .map_err(|e| format!("Speech model discovery task failed: {e}"))?
}

#[tauri::command]
pub async fn dictation_settings(app: tauri::AppHandle) -> Result<Settings, String> {
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let _storage = STORAGE
            .lock()
            .map_err(|_| "Dictation settings lock failed.")?;
        load(&directory)
    })
    .await
    .map_err(|e| format!("Dictation settings task failed: {e}"))?
}

#[tauri::command]
pub async fn dictation_set_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, DictationState>,
    model_path: String,
) -> Result<Settings, String> {
    validate_path(&model_path)?;
    let guard = state
        .0
        .clone()
        .try_lock_owned()
        .map_err(|_| "Dictation is busy. Wait before changing the speech model.")?;
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        let _storage = STORAGE
            .lock()
            .map_err(|_| "Dictation settings lock failed.")?;
        save(&directory, &model_path)
    })
    .await
    .map_err(|e| format!("Dictation settings task failed: {e}"))?
}

fn validate_speech_plan(plan: &lma::fetch::huggingface::InstallPlan) -> Result<(), String> {
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        if !plan.files.iter().any(|asset| asset.name == name) {
            return Err(format!(
                "This dictation backend requires {name} at the repository root; sharded weights and GGML/GGUF files are not supported."
            ));
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn dictation_model_plan(
    repository: String,
) -> Result<lma::fetch::huggingface::InstallPlan, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let plan = lma::fetch::huggingface::inspect(&repository, None, None)?;
        validate_speech_plan(&plan)?;
        Ok(plan)
    })
    .await
    .map_err(|e| format!("Model inspection task failed: {e}"))?
}

#[tauri::command]
pub async fn dictation_install_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, DictationState>,
    plan: lma::fetch::huggingface::InstallPlan,
) -> Result<Settings, String> {
    validate_speech_plan(&plan)?;
    let guard = state
        .0
        .clone()
        .try_lock_owned()
        .map_err(|_| "Dictation is busy. Wait before installing the speech model.")?;
    let config = app.path().app_config_dir().map_err(|e| e.to_string())?;

    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        {
            let _storage = STORAGE
                .lock()
                .map_err(|_| "Dictation settings lock failed.")?;
            load(&config)?;
        }
        // Only this explicit command may download weights. Do not hold the
        // settings lock during network I/O or send microphone samples to it.
        let installed = lma::fetch::huggingface::install(&plan)?;
        lma::speech::validate_model_directory(&installed)?;
        let _storage = STORAGE
            .lock()
            .map_err(|_| "Dictation settings lock failed.")?;
        save(
            &config,
            installed
                .to_str()
                .ok_or("Whisper model path must be UTF-8.")?,
        )
    })
    .await
    .map_err(|e| format!("Speech model installation task failed: {e}"))?
}

#[derive(Serialize)]
pub struct Text {
    text: String,
    language: Option<String>,
}

#[tauri::command]
pub async fn dictation_transcribe(
    app: tauri::AppHandle,
    state: tauri::State<'_, DictationState>,
    pcm16: String,
) -> Result<Text, String> {
    if pcm16.is_empty() || pcm16.len() > MAX_ENCODED {
        return Err("Dictation audio exceeds the 120-second limit or is empty.".into());
    }
    let guard = state
        .0
        .clone()
        .try_lock_owned()
        .map_err(|_| "Dictation is busy installing or transcribing. Wait for it to finish.")?;
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    // The blocking task owns the lock even if the webview disconnects. Inference
    // cannot be cancelled by dropping an IPC future; never allow it to overlap.
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        let settings = {
            let _storage = STORAGE
                .lock()
                .map_err(|_| "Dictation settings lock failed.")?;
            load(&directory)?
        };
        let model = model_path(settings.model_path.as_deref().ok_or(
            "Install the speech model or choose a local Hugging Face Whisper directory first.",
        )?)?;
        let samples = decode_pcm(&pcm16)?;
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8);
        let transcript = lma::speech::transcribe_samples(
            &model,
            &samples,
            None,
            threads,
            lma::speech::gpu_available(),
        )?;
        Ok(Text {
            text: transcript.text,
            language: transcript.language,
        })
    })
    .await
    .map_err(|e| format!("Dictation task failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "lma-dictation-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&directory).unwrap();
            Self(fs::canonicalize(directory).unwrap())
        }
        fn model(&self) -> PathBuf {
            let path = self.0.join("speech-model");
            fs::create_dir(&path).unwrap();
            fs::write(
                path.join("config.json"),
                br#"{"model_type":"whisper","name":"Fixture model"}"#,
            )
            .unwrap();
            for asset in ["tokenizer.json", "model.safetensors"] {
                fs::write(path.join(asset), b"fixture").unwrap();
            }
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn pcm_is_bounded_strict_little_endian_and_normalized() {
        assert_eq!(
            decode_pcm("AIAAAP9/").unwrap(),
            [-1.0, 0.0, 32767.0 / 32768.0]
        );
        for bad in ["", "!!!!", "AA==", "AAA=\n"] {
            assert!(decode_pcm(bad).is_err());
        }
        assert_eq!(
            decode_pcm(&STANDARD.encode(vec![0; MAX_SAMPLES * 2]))
                .unwrap()
                .len(),
            MAX_SAMPLES
        );
        assert!(decode_pcm(&STANDARD.encode(vec![0; MAX_SAMPLES * 2 + 2])).is_err());
    }
    #[test]
    fn settings_only_remember_model_privately_and_preserve_invalid_files() {
        let fixture = Fixture::new();
        let config = fixture.0.join("config");
        assert!(load(&config).unwrap().model_path.is_none());
        assert!(!config.exists());
        let model = fixture.model();
        save(&config, model.to_str().unwrap()).unwrap();
        save(&config, model.to_str().unwrap()).unwrap();
        assert_eq!(load(&config).unwrap().model_path.as_deref(), model.to_str());
        assert_eq!(fs::read_dir(&config).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(config.join(FILE_NAME))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(config.join(FILE_NAME)).unwrap()).unwrap();
        assert_eq!(value["settings"].as_object().unwrap().len(), 1);
        for bad in [
            b"{broken".to_vec(),
            vec![b' '; MAX_SETTINGS as usize + 1],
            br#"{"version":2,"settings":{"modelPath":null}}"#.to_vec(),
            br#"{"version":1,"settings":{"modelPath":"relative.bin"}}"#.to_vec(),
            br#"{"version":1,"settings":{"modelPath":null,"key":"private"}}"#.to_vec(),
        ] {
            fs::write(config.join(FILE_NAME), &bad).unwrap();
            assert!(load(&config).is_err());
            assert!(save(&config, model.to_str().unwrap()).is_err());
            assert_eq!(fs::read(config.join(FILE_NAME)).unwrap(), bad);
        }
    }
    #[test]
    fn rejects_wrong_model_and_stale_models_remain_recoverable() {
        let fixture = Fixture::new();
        assert!(model_path("relative.bin").is_err());
        assert!(model_path(fixture.0.to_str().unwrap()).is_err());
        let ggml = fixture.0.join("ggml-tiny.bin");
        fs::write(&ggml, b"ggml").unwrap();
        assert!(
            model_path(ggml.to_str().unwrap())
                .unwrap_err()
                .contains("directory")
        );
        let model = fixture.model();
        save(&fixture.0, model.to_str().unwrap()).unwrap();
        fs::remove_file(model.join("config.json")).unwrap();
        assert!(model_path(model.to_str().unwrap()).is_err());
        fs::remove_dir_all(&model).unwrap();
        assert!(load(&fixture.0).unwrap().model_path.is_some());
    }
    #[cfg(unix)]
    #[test]
    fn settings_symlinks_are_never_followed_or_replaced() {
        let fixture = Fixture::new();
        let model = fixture.model();
        let outside = fixture.0.join("outside");
        fs::write(&outside, b"keep").unwrap();
        std::os::unix::fs::symlink(&outside, fixture.0.join(FILE_NAME)).unwrap();
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, model.to_str().unwrap()).is_err());
        assert_eq!(fs::read(outside).unwrap(), b"keep");
    }
    #[test]
    fn discovery_only_lists_complete_whisper_models_and_uses_real_metadata_names() {
        let fixture = Fixture::new();
        let model = fixture.model();
        let roots = [fixture.0.clone(), model.clone()];
        let catalog = scan_models(&roots, MAX_MODEL_ENTRIES);
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].name, "Fixture model");
        assert_eq!(catalog.models[0].model_path, model.to_str().unwrap());
        fs::write(
            model.join(".lma-source.json"),
            br#"{"repository":"fixture-owner/speech-assets"}"#,
        )
        .unwrap();
        assert_eq!(
            scan_models(&roots, MAX_MODEL_ENTRIES).models[0].name,
            "fixture-owner/speech-assets"
        );
        fs::remove_file(model.join(".lma-source.json")).unwrap();
        fs::write(model.join("config.json"), br#"{"model_type":"other"}"#).unwrap();
        assert!(scan_models(&roots, MAX_MODEL_ENTRIES).models.is_empty());
        assert!(
            model_path(model.to_str().unwrap())
                .unwrap_err()
                .contains("model_type: whisper")
        );
        fs::write(model.join("config.json"), br#"{"model_type":"whisper"}"#).unwrap();
        assert_eq!(
            scan_models(&roots, MAX_MODEL_ENTRIES).models[0].name,
            model.to_str().unwrap()
        );
        fs::remove_file(model.join("tokenizer.json")).unwrap();
        assert!(scan_models(&roots, MAX_MODEL_ENTRIES).models.is_empty());
    }

    #[test]
    fn discovery_bounds_entries_depth_metadata_and_skips_install_staging_directories() {
        let fixture = Fixture::new();
        let model = fixture.model();
        let limited = scan_models(&[fixture.0.clone()], 1);
        assert!(limited.models.is_empty());
        assert!(!limited.warnings.is_empty());
        fs::write(
            model.join("config.json"),
            vec![b' '; MAX_MODEL_JSON as usize + 1],
        )
        .unwrap();
        assert!(
            scan_models(&[fixture.0.clone()], MAX_MODEL_ENTRIES)
                .models
                .is_empty()
        );
        fs::write(model.join("config.json"), br#"{"model_type":"whisper"}"#).unwrap();
        let staging = fixture.0.join(".model-install-staging");
        fs::rename(&model, &staging).unwrap();
        assert!(
            scan_models(&[fixture.0.clone()], MAX_MODEL_ENTRIES)
                .models
                .is_empty()
        );
        let mut deep = fixture.0.clone();
        for _ in 0..=MAX_MODEL_DEPTH {
            deep = deep.join("nested");
        }
        fs::create_dir_all(&deep).unwrap();
        fs::rename(staging, deep.join("speech-model")).unwrap();
        let catalog = scan_models(&[fixture.0.clone()], MAX_MODEL_ENTRIES);
        assert!(catalog.models.is_empty());
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("depth limit"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn discovery_does_not_follow_nested_directory_or_metadata_symlinks() {
        let fixture = Fixture::new();
        let outside = Fixture::new();
        let model = outside.model();
        std::os::unix::fs::symlink(&model, fixture.0.join("linked-model")).unwrap();
        assert!(
            scan_models(&[fixture.0.clone()], MAX_MODEL_ENTRIES)
                .models
                .is_empty()
        );
        let own = fixture.model();
        fs::remove_file(own.join("config.json")).unwrap();
        std::os::unix::fs::symlink(model.join("config.json"), own.join("config.json")).unwrap();
        assert!(
            scan_models(&[fixture.0.clone()], MAX_MODEL_ENTRIES)
                .models
                .is_empty()
        );
    }

    #[test]
    fn inference_guard_survives_the_request_owner() {
        let state = DictationState::default();
        let guard = state.0.clone().try_lock_owned().unwrap();
        assert!(state.0.clone().try_lock_owned().is_err());
        drop(guard);
        assert!(state.0.clone().try_lock_owned().is_ok());
    }
}
