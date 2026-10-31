//! Desktop settings and the deliberately narrow defaults-config adapter.
//! Defaults JSON accepts only `gpu` (auto/all/cpu/layer count), `gpu_layers`
//! (CLI-serialized u32, mutually exclusive with gpu), and `context` (u32 >= 2).
//! Missing fields use ServeConfig defaults. No other ServeConfig fields, paths,
//! commands, environment expansion, credentials, or fallback policy are applied.
use crate::state::LoadOptions;
use lma::config::ServeConfig;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const FILE_NAME: &str = "app-settings.json";
const MAX_BYTES: u64 = 64 * 1024;
static STORAGE: Mutex<()> = Mutex::new(());
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LoadPolicy {
    #[default]
    Automatic,
    Defaults,
    Manual,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct Settings {
    pub load_policy: LoadPolicy,
    pub defaults_path: Option<String>,
    pub auto_restore: bool,
    pub gpu: String,
    pub context: u32,
    pub model_directories: Vec<String>,
    pub ollama_directory: Option<String>,
    pub lm_studio_directories: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            load_policy: LoadPolicy::Automatic,
            defaults_path: None,
            auto_restore: true,
            gpu: "auto".into(),
            context: ServeConfig::default().context,
            model_directories: Vec::new(),
            ollama_directory: None,
            lm_studio_directories: Vec::new(),
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        gpu(&self.gpu)?;
        context(self.context)?;
        if self.model_directories.len() + self.lm_studio_directories.len() > 64 {
            return Err("At most 64 custom model directories are supported.".into());
        }
        for value in self
            .model_directories
            .iter()
            .chain(&self.lm_studio_directories)
            .chain(self.defaults_path.iter())
            .chain(self.ollama_directory.iter())
        {
            path(value)?;
        }
        Ok(())
    }

    /// Transform only runtime options; preserve selected model/store/projector IDs.
    pub fn apply(&self, mut options: LoadOptions) -> Result<LoadOptions, String> {
        self.validate()?;
        match self.load_policy {
            LoadPolicy::Automatic => {
                options.gpu = "auto".into();
                options.context = ServeConfig::default().context;
            }
            LoadPolicy::Manual => {}
            LoadPolicy::Defaults => {
                let location = match &self.defaults_path {
                    Some(value) => path(value)?,
                    None => lma::models::discovery::default_models_dir()?
                        .parent()
                        .ok_or("Cannot determine the default config directory.")?
                        .join("config.json"),
                };
                let bytes = read_regular(&location)?;
                let defaults: DefaultsConfig = serde_json::from_slice(&bytes)
                    .map_err(|_| "Defaults config must contain only gpu or gpu_layers, and context, with valid JSON types.".to_string())?;
                let (gpu, context) = defaults.options()?;
                options.gpu = gpu;
                options.context = context;
            }
        }
        options.clone().config()?;
        Ok(options)
    }
}

pub fn path(value: &str) -> Result<PathBuf, String> {
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(
            "Settings paths must be nonempty and at most 4096 bytes without control characters."
                .into(),
        );
    }
    let result = if let Some(relative) = value.strip_prefix("~/") {
        let models = lma::models::discovery::default_models_dir()?;
        models
            .parent()
            .and_then(Path::parent)
            .ok_or("Cannot determine the home directory.")?
            .join(relative)
    } else {
        PathBuf::from(value)
    };
    if !result.is_absolute() {
        return Err("Settings paths must be absolute (or start with ~/).".into());
    }
    Ok(result)
}

fn gpu(value: &str) -> Result<(), String> {
    if matches!(value, "auto" | "all" | "cpu")
        || (!value.is_empty()
            && value.bytes().all(|v| v.is_ascii_digit())
            && value.parse::<u32>().is_ok())
    {
        Ok(())
    } else {
        Err("GPU must be auto, all, cpu, or an unsigned layer count.".into())
    }
}

fn context(value: u32) -> Result<(), String> {
    if value < 2 {
        Err("Context must be at least 2 tokens.".into())
    } else {
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ConfigGpu {
    Text(String),
    Layers(u32),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefaultsConfig {
    #[serde(default, deserialize_with = "present")]
    gpu: Option<ConfigGpu>,
    #[serde(default, deserialize_with = "present")]
    gpu_layers: Option<u32>,
    #[serde(default, deserialize_with = "present")]
    context: Option<u32>,
}
// An absent field uses the shared default; an explicit null is an invalid value.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}
impl DefaultsConfig {
    fn options(self) -> Result<(String, u32), String> {
        if self.gpu.is_some() && self.gpu_layers.is_some() {
            return Err("Defaults config must not specify both gpu and gpu_layers.".into());
        }
        let gpu_value = match (self.gpu, self.gpu_layers) {
            (Some(ConfigGpu::Text(value)), _) => value,
            (Some(ConfigGpu::Layers(value)), _) | (_, Some(value)) => value.to_string(),
            _ => "auto".into(),
        };
        gpu(&gpu_value)?;
        let context_value = self
            .context
            .unwrap_or_else(|| ServeConfig::default().context);
        context(context_value)?;
        Ok((gpu_value, context_value))
    }
}

pub fn load(config_dir: &Path) -> Result<Settings, String> {
    let _guard = STORAGE
        .lock()
        .map_err(|_| "Settings storage lock failed.")?;
    load_inner(config_dir)
}

fn load_inner(config_dir: &Path) -> Result<Settings, String> {
    let location = lma::models::storage::read_path(config_dir, FILE_NAME)?;
    match fs::symlink_metadata(&location) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Settings::default()),
        Err(e) => return Err(format!("Cannot inspect app settings: {e}")),
        Ok(_) => {}
    }
    let bytes = read_regular(&location)?;
    let settings: Settings = serde_json::from_slice(&bytes)
        .map_err(|_| "App settings contain invalid or unsupported JSON.".to_string())?;
    settings.validate()?;
    Ok(settings)
}

pub fn save(config_dir: &Path, settings: Settings) -> Result<(), String> {
    let _guard = STORAGE
        .lock()
        .map_err(|_| "Settings storage lock failed.")?;
    settings.validate()?;
    let bytes = serde_json::to_vec_pretty(&settings).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("App settings exceed 64 KiB.".into());
    }
    // An invalid/unknown/redirected existing file requires explicit user repair.
    load_inner(config_dir)?;
    fs::create_dir_all(config_dir).map_err(|e| e.to_string())?;
    let temporary = Temporary(config_dir.join(format!(
        ".app-settings-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    let mut flags = OpenOptions::new();
    flags.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        flags.mode(0o600);
    }
    let mut file = flags.open(&temporary.0).map_err(|e| e.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    load_inner(config_dir)?;
    fs::rename(&temporary.0, config_dir.join(FILE_NAME)).map_err(|e| e.to_string())?;
    Ok(())
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn read_regular(path: &Path) -> Result<Vec<u8>, String> {
    let before =
        fs::symlink_metadata(path).map_err(|e| format!("Cannot inspect settings/config: {e}"))?;
    if !before.file_type().is_file() || before.len() > MAX_BYTES {
        return Err(
            "Settings/config must be a regular, non-symlink file of at most 64 KiB.".into(),
        );
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800); // O_NOFOLLOW | O_NONBLOCK
    }
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    let file: File = options
        .open(path)
        .map_err(|e| format!("Cannot open settings/config: {e}"))?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if !after.file_type().is_file() || after.len() > MAX_BYTES {
        return Err("Settings/config must be a regular file of at most 64 KiB.".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Err("Settings/config changed while opening it.".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("Settings/config exceed 64 KiB.".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lma-settings-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(fs::canonicalize(path).unwrap())
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn options() -> LoadOptions {
        LoadOptions {
            model: "/fixture/model.gguf".into(),
            ollama: false,
            ollama_dir: None,
            projector: None,
            gpu: "cpu".into(),
            context: 128,
        }
    }

    #[test]
    fn defaults_payload_and_atomic_private_roundtrip() {
        let dir = Fixture::new();
        let missing = dir.path().join("missing");
        let settings = load(&missing).unwrap();
        assert!(!missing.exists());
        assert_eq!(
            serde_json::to_value(&settings).unwrap(),
            serde_json::json!({
                "loadPolicy":"automatic", "defaultsPath":null, "autoRestore":true,
                "gpu":"auto", "context":4096, "modelDirectories":[],
                "ollamaDirectory":null, "lmStudioDirectories":[]
            })
        );
        save(&missing, settings.clone()).unwrap();
        let mut changed = settings;
        changed.auto_restore = false;
        changed.load_policy = LoadPolicy::Manual;
        changed.model_directories.push("/fixture/custom".into());
        save(&missing, changed.clone()).unwrap();
        assert_eq!(load(&missing).unwrap(), changed);
        assert_eq!(fs::read_dir(&missing).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(missing.join(FILE_NAME))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn corrupt_oversized_unknown_invalid_and_symlink_settings_are_preserved() {
        let dir = Fixture::new();
        let location = dir.path().join(FILE_NAME);
        for bytes in [
            b"{broken".to_vec(),
            vec![b' '; MAX_BYTES as usize + 1],
            br#"{"secret":"do not retain"}"#.to_vec(),
            br#"{"context":0}"#.to_vec(),
            br#"{"loadPolicy":"other"}"#.to_vec(),
            br#"{"modelDirectories":["relative"]}"#.to_vec(),
        ] {
            fs::write(&location, &bytes).unwrap();
            assert!(load(dir.path()).is_err());
            assert!(save(dir.path(), Settings::default()).is_err());
            assert_eq!(fs::read(&location).unwrap(), bytes);
        }
        #[cfg(unix)]
        {
            fs::remove_file(&location).unwrap();
            let target = dir.path().join("target");
            fs::write(&target, b"{}").unwrap();
            std::os::unix::fs::symlink(&target, &location).unwrap();
            assert!(load(dir.path()).is_err());
            assert!(save(dir.path(), Settings::default()).is_err());
            assert!(
                fs::symlink_metadata(&location)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            assert_eq!(fs::read(&target).unwrap(), b"{}");
        }
    }

    #[test]
    fn policies_apply_only_gpu_and_context_without_changing_selection() {
        let mut settings = Settings::default();
        let automatic = settings.apply(options()).unwrap();
        assert_eq!(
            automatic.config().unwrap().gpu_layers,
            lma::config::default_gpu_layers()
        );
        assert_eq!(
            settings.apply(options()).unwrap().context,
            ServeConfig::default().context
        );
        settings.load_policy = LoadPolicy::Manual;
        assert_eq!(settings.apply(options()).unwrap(), options());
        let dir = Fixture::new();
        let config = dir.path().join("config.json");
        settings.defaults_path = Some(config.to_str().unwrap().into());
        settings.load_policy = LoadPolicy::Defaults;
        for (json, gpu, context) in [
            (r#"{}"#, "auto", 4096),
            (r#"{"gpu":"all","context":8192}"#, "all", 8192),
            (r#"{"gpu_layers":0,"context":256}"#, "0", 256),
            (r#"{"gpu":12}"#, "12", 4096),
        ] {
            fs::write(&config, json).unwrap();
            let actual = settings.apply(options()).unwrap();
            assert_eq!(actual.model, options().model);
            assert_eq!(actual.gpu, gpu);
            assert_eq!(actual.context, context);
            assert_eq!(actual.config().unwrap().memory_fallback, "error");
        }
        for json in [
            r#"{"gpu":"bad"}"#,
            r#"{"gpu":-1}"#,
            r#"{"context":1}"#,
            r#"{"gpu":"cpu","gpu_layers":1}"#,
            r#"{"gpu_layers":1.5}"#,
            r#"{"context":4294967296}"#,
            r#"{"gpu":null}"#,
            r#"{"context":null}"#,
            r#"{"gpu_layers":null}"#,
            r#"{"model":"/do/not/execute"}"#,
            r#"{"api_key":"secret"}"#,
            r#"{"memory_fallback":"cpu"}"#,
        ] {
            fs::write(&config, json).unwrap();
            assert!(settings.apply(options()).is_err(), "{json}");
            assert_eq!(fs::read_to_string(&config).unwrap(), json);
        }
        fs::remove_file(config).unwrap();
        assert!(settings.apply(options()).is_err());
    }
}
