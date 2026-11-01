//! Explicit, user-initiated local source updates, not a signed-release updater.
//! The caller must unload inference before `start` and gate it while `is_running`.
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};

const LOG_LIMIT: usize = 64 * 1024;
const CONFIG_NAME: &str = "source-install.json";

#[derive(Clone, Serialize)]
pub struct UpdateStatus {
    pub state: String,
    pub log: String,
    pub error: Option<String>,
}

struct Progress {
    running: bool,
    state: &'static str,
    log: Vec<u8>,
    error: Option<String>,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            running: false,
            state: "Idle",
            log: Vec::new(),
            error: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct SourceUpdater {
    inner: Arc<Mutex<Progress>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceConfig {
    schema_version: u32,
    source_dir: PathBuf,
    backend: String,
    jobs: u32,
    #[serde(default, deserialize_with = "deserialize_desktop")]
    desktop: Option<bool>,
    install_locations: HashMap<String, PathBuf>,
}

// Missing means legacy both-component metadata; an explicit null is not a bool.
fn deserialize_desktop<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<bool>, D::Error> {
    bool::deserialize(deserializer).map(Some)
}

fn regular_file(path: &Path) -> Result<(), String> {
    let meta = fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.file_type().is_file() {
        return Err(format!(
            "Expected a regular, non-symlink file: {}",
            path.display()
        ));
    }
    Ok(())
}

fn directory(path: &Path) -> Result<(), String> {
    let meta = fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !path.is_absolute() || !meta.file_type().is_dir() {
        return Err(format!(
            "Expected an absolute, non-symlink directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn read_config(config_dir: &Path) -> Result<SourceConfig, String> {
    let path = lma::models::storage::read_path(config_dir, CONFIG_NAME)?;
    directory(
        path.parent()
            .ok_or("Source registration has no parent directory")?,
    )?;
    regular_file(&path)
        .map_err(|e| format!("No safe source registration. Run install/install.sh first. {e}"))?;
    let mut data = Vec::new();
    fs::File::open(&path)
        .map_err(|e| e.to_string())?
        .take((LOG_LIMIT + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > LOG_LIMIT {
        return Err("Source registration exceeds 64 KiB".into());
    }
    let config: SourceConfig =
        serde_json::from_slice(&data).map_err(|e| format!("Invalid source registration: {e}"))?;
    if config.schema_version != 1 || !(1..=64).contains(&config.jobs) {
        return Err("Unsupported source registration schema or jobs (expected 1..64)".into());
    }
    let backend_ok = if cfg!(target_os = "macos") {
        config.backend == "metal"
            || (config.desktop == Some(false) && matches!(config.backend.as_str(), "cpu" | "cuda"))
    } else {
        matches!(config.backend.as_str(), "cpu" | "cuda")
    };
    if !backend_ok {
        return Err(
            "Invalid backend for this platform (macOS requires metal; Linux/Windows cpu or cuda)"
                .into(),
        );
    }
    directory(&config.source_dir)?;
    // The installer registers a canonical path; disallow a redirected checkout.
    if fs::canonicalize(&config.source_dir).map_err(|e| e.to_string())? != config.source_dir {
        return Err(
            "Registered source path is not canonical; reinstall from the intended checkout".into(),
        );
    }
    if !config.install_locations.contains_key("cli")
        || (config.desktop != Some(false) && !config.install_locations.contains_key("desktop"))
        || config.install_locations.values().any(|p| !p.is_absolute())
    {
        return Err(
            "Source registration requires absolute cli and, unless desktop is false, desktop install locations".into(),
        );
    }
    directory(&config.source_dir.join("install"))?;
    directory(&config.source_dir.join("desktop"))?;
    for relative in [
        "Cargo.toml",
        "Cargo.lock",
        "desktop/Cargo.toml",
        "install/install.sh",
        "install/source_install.py",
    ] {
        regular_file(&config.source_dir.join(relative))?;
    }
    Ok(config)
}

fn validate_update_config(config_dir: &Path) -> Result<SourceConfig, String> {
    if cfg!(target_os = "windows") {
        Err("Windows source updates are external: close lma-desktop and lma, then run install/install.ps1 --update --desktop in your registered checkout. If binaries are staged, close the app and run --apply-staged. Restart manually.".into())
    } else if cfg!(any(target_os = "macos", target_os = "linux")) {
        read_config(config_dir)
    } else {
        Err("Source updates are not supported on this platform".into())
    }
}

// Invalid UTF-8 is represented as \xNN, not silently replaced or dropped.
fn display_bytes(mut bytes: &[u8]) -> String {
    let mut text = String::new();
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                text.push_str(valid);
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                text.push_str(std::str::from_utf8(&bytes[..valid]).expect("valid prefix"));
                let end = valid + error.error_len().unwrap_or(bytes.len() - valid);
                for byte in &bytes[valid..end] {
                    use std::fmt::Write;
                    let _ = write!(text, "\\x{byte:02x}");
                }
                bytes = &bytes[end..];
            }
        }
    }
    if text.len() > LOG_LIMIT {
        let mut start = text.len() - LOG_LIMIT;
        while !text.is_char_boundary(start) {
            start += 1;
        }
        text.drain(..start);
    }
    text
}

impl SourceUpdater {
    pub fn is_running(&self) -> bool {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).running
    }

    pub fn status(&self) -> UpdateStatus {
        let state = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        UpdateStatus {
            state: state.state.into(),
            log: display_bytes(&state.log),
            error: state.error.clone(),
        }
    }

    /// Non-destructive platform/registration checks before the host unloads a
    /// model. Startup repeats validation to detect intervening local changes.
    pub fn preflight(&self, config_dir: &Path) -> Result<(), String> {
        if self.is_running() {
            return Err("A source update is already running".into());
        }
        validate_update_config(config_dir).map(|_| ())
    }

    pub fn start(&self, config_dir: PathBuf) -> Result<String, String> {
        let mut state = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if state.running {
            return Err("A source update is already running".into());
        }
        let validation = validate_update_config(&config_dir);
        let config = match validation {
            Ok(config) => config,
            Err(error) => {
                state.state = "Failed";
                state.error = Some(error.clone());
                state.log.clear();
                return Err(error);
            }
        };
        state.running = true;
        state.state = "Running";
        state.error = None;
        state.log.clear();
        state
            .log
            .extend_from_slice(b"User-requested local source build; not a signed release.\n");
        let updater = self.clone();
        let spawn = thread::Builder::new()
            .name("lma-source-update".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    updater.execute(config, config_dir)
                }))
                .unwrap_or_else(|_| Err("Source update worker panicked".into()));
                let mut state = updater.inner.lock().unwrap_or_else(|p| p.into_inner());
                state.running = false;
                match result {
                    Ok(()) => {
                        state.state = "Completed";
                        append(
                            &mut state.log,
                            b"\nCompleted. Restart the app manually to use the new build.\n",
                        );
                    }
                    Err(error) => {
                        state.state = "Failed";
                        state.error = Some(error);
                    }
                }
            });
        if let Err(error) = spawn {
            state.running = false;
            state.state = "Failed";
            let error = format!("Cannot start source update worker: {error}");
            state.error = Some(error.clone());
            return Err(error);
        }
        Ok("Source update started. Builds may take several minutes; restart manually when Completed.".into())
    }

    fn execute(&self, config: SourceConfig, config_dir: PathBuf) -> Result<(), String> {
        // Finder does not source shell profiles. Preserve existing precedence, then
        // add common user-managed tool locations without executing a login shell.
        let inherited = std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into());
        let mut paths: Vec<PathBuf> = std::env::split_paths(&inherited).collect();
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            paths.extend([home.join(".cargo/bin"), home.join(".local/bin")]);
        }
        paths.push(PathBuf::from("/usr/local/bin"));
        if cfg!(target_os = "macos") {
            paths.push(PathBuf::from("/opt/homebrew/bin"));
        }
        let path = std::env::join_paths(paths).map_err(|e| format!("Invalid build PATH: {e}"))?;
        let mut child = Command::new("sh")
            .arg(config.source_dir.join("install/install.sh"))
            .arg("--update")
            .arg("--desktop")
            // A CLI-only macOS registration can select CPU, but the GUI needs Metal.
            .arg("--backend").arg(if cfg!(target_os = "macos") { "metal".into() } else { config.backend })
            .arg("--jobs").arg(config.jobs.to_string())
            .arg("--config-dir").arg(config_dir)
            .current_dir(config.source_dir)
            .env("PATH", path)
            .env("NO_COLOR", "1").env("CARGO_TERM_COLOR", "never")
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().map_err(|e| format!("Cannot launch source installer: {e}. Check Python, Rust and build prerequisites; GUI apps may have a restricted PATH."))?;
        let stdout = child.stdout.take().ok_or("Installer stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Installer stderr unavailable")?;
        // Drain both pipes concurrently so compiler output cannot deadlock the child.
        thread::scope(|scope| {
            let out = scope.spawn(|| self.read_output(stdout));
            let err = scope.spawn(|| self.read_output(stderr));
            let exit = child
                .wait()
                .map_err(|e| format!("Cannot wait for installer: {e}"));
            let out = out
                .join()
                .map_err(|_| "stdout reader panicked".to_string())?;
            let err = err
                .join()
                .map_err(|_| "stderr reader panicked".to_string())?;
            out?;
            err?;
            let exit = exit?;
            if exit.success() {
                Ok(())
            } else {
                Err(format!(
                    "Source installer exited with {exit}. See the update log; the app has not been restarted."
                ))
            }
        })
    }

    fn read_output(&self, mut reader: impl Read) -> Result<(), String> {
        let mut buffer = [0; 4096];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => return Ok(()),
                Ok(size) => {
                    let mut state = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                    append(&mut state.log, &buffer[..size]);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(format!("Cannot read installer output: {error}")),
            }
        }
    }
}

fn append(log: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.len() >= LOG_LIMIT {
        log.clear();
        log.extend_from_slice(&bytes[bytes.len() - LOG_LIMIT..]);
    } else {
        let excess = (log.len() + bytes.len()).saturating_sub(LOG_LIMIT);
        log.drain(..excess);
        log.extend_from_slice(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(script: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "lma-update-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(path.join("install")).unwrap();
            fs::create_dir_all(path.join("desktop")).unwrap();
            fs::create_dir_all(path.join("config")).unwrap();
            let path = fs::canonicalize(path).unwrap();
            for name in [
                "Cargo.toml",
                "Cargo.lock",
                "desktop/Cargo.toml",
                "install/source_install.py",
            ] {
                fs::write(path.join(name), "test fixture").unwrap();
            }
            fs::write(path.join("install/install.sh"), script).unwrap();
            let config = serde_json::json!({
                "schema_version": 1, "source_dir": path,
                "backend": if cfg!(target_os = "macos") { "metal" } else { "cpu" }, "jobs": 2,
                "install_locations": { "cli": path.join("bin/lma"), "desktop": path.join("bin/desktop") }
            });
            fs::write(
                path.join("config").join(CONFIG_NAME),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            Self(path)
        }
        fn components(&self, desktop: Option<bool>) {
            let path = self.config().join(CONFIG_NAME);
            let mut config: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            if let Some(desktop) = desktop {
                config["desktop"] = serde_json::json!(desktop);
                if !desktop {
                    config["install_locations"]
                        .as_object_mut()
                        .unwrap()
                        .remove("desktop");
                }
            }
            fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
        }
        fn config(&self) -> PathBuf {
            self.0.join("config")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn byte_log_is_bounded_and_lossless_for_invalid_utf8() {
        let mut log = Vec::new();
        append(&mut log, &vec![b'a'; LOG_LIMIT + 10]);
        append(&mut log, b"\xff\xe2");
        append(&mut log, b"\x82\xac");
        assert_eq!(log.len(), LOG_LIMIT);
        let display = display_bytes(&log);
        assert!(display.ends_with("\\xff€"));
        assert!(display.len() <= LOG_LIMIT);
    }

    #[test]
    fn validates_config_and_exposes_errors() {
        let fixture = Fixture::new("exit 0\n");
        assert!(read_config(&fixture.config()).is_ok());
        fs::write(fixture.config().join(CONFIG_NAME), b"{broken}").unwrap();
        assert!(read_config(&fixture.config()).is_err());
        let updater = SourceUpdater::default();
        assert!(updater.start(fixture.config()).is_err());
        assert_eq!(updater.status().state, "Failed");
        assert!(updater.status().error.is_some());
        assert!(!updater.is_running());
    }

    #[test]
    fn validates_legacy_both_and_cli_only_registrations() {
        for desktop in [None, Some(true), Some(false)] {
            let fixture = Fixture::new("exit 0\n");
            fixture.components(desktop);
            let config = read_config(&fixture.config()).unwrap();
            assert_eq!(config.desktop, desktop);
            let path = fixture.config().join(CONFIG_NAME);
            let original: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            for key in ["cli", "desktop"] {
                for value in [None, Some(serde_json::json!("relative"))] {
                    let mut changed = original.clone();
                    let locations = changed["install_locations"].as_object_mut().unwrap();
                    if let Some(value) = value {
                        locations.insert(key.into(), value);
                    } else {
                        locations.remove(key);
                    }
                    fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
                    assert_eq!(
                        read_config(&fixture.config()).is_ok(),
                        desktop == Some(false)
                            && key == "desktop"
                            && !changed["install_locations"]
                                .as_object()
                                .unwrap()
                                .contains_key(key),
                        "desktop={desktop:?}, invalid location={key}: {changed}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_schema_jobs_paths_and_unknown_fields() {
        let fixture = Fixture::new("exit 0\n");
        let path = fixture.config().join(CONFIG_NAME);
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for (key, value) in [
            ("schema_version", serde_json::json!(2)),
            ("jobs", serde_json::json!(0)),
            ("jobs", serde_json::json!(65)),
            ("backend", serde_json::json!("arbitrary")),
            ("source_dir", serde_json::json!("relative/path")),
            ("command", serde_json::json!("not allowed")),
            ("desktop", serde_json::json!(null)),
            ("desktop", serde_json::json!(0)),
            ("desktop", serde_json::json!(1)),
            ("desktop", serde_json::json!("false")),
            ("desktop", serde_json::json!([])),
            ("desktop", serde_json::json!({})),
            (
                "install_locations",
                serde_json::json!({"cli": "relative", "desktop": "relative"}),
            ),
        ] {
            let mut changed = original.clone();
            changed[key] = value;
            fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(
                read_config(&fixture.config()).is_err(),
                "accepted invalid {key}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_registration_and_installer() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new("exit 0\n");
        let config = fixture.config().join(CONFIG_NAME);
        fs::rename(&config, fixture.0.join("saved.json")).unwrap();
        symlink(fixture.0.join("saved.json"), &config).unwrap();
        assert!(read_config(&fixture.config()).is_err());
        fs::remove_file(&config).unwrap();
        fs::rename(fixture.0.join("saved.json"), &config).unwrap();
        fs::remove_file(fixture.0.join("install/install.sh")).unwrap();
        symlink("source_install.py", fixture.0.join("install/install.sh")).unwrap();
        assert!(read_config(&fixture.config()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn one_active_update_across_clones_and_visible_exit_failure() {
        let fixture = Fixture::new(
            "printf '\\377hello\\n'; printf 'stderr\\n' >&2; while [ ! -f release ]; do sleep 0.01; done; exit 7\n",
        );
        let updater = SourceUpdater::default();
        updater.start(fixture.config()).unwrap();
        assert!(updater.is_running());
        assert!(updater.clone().start(fixture.config()).is_err());
        fs::write(fixture.0.join("release"), "go").unwrap();
        wait(&updater);
        let status = updater.status();
        assert_eq!(status.state, "Failed");
        assert!(status.error.unwrap().contains('7'));
        assert!(status.log.contains("\\xffhello"));
        assert!(status.log.contains("stderr"));
        fs::write(fixture.0.join("install/install.sh"), "exit 0\n").unwrap();
        updater.start(fixture.config()).unwrap();
        wait(&updater);
        assert_eq!(updater.status().state, "Completed");
        assert!(updater.status().log.contains("Restart"));
    }

    #[cfg(unix)]
    #[test]
    fn explicit_gui_update_always_requests_desktop() {
        for desktop in [None, Some(true), Some(false)] {
            let fixture = Fixture::new("printf '%s\\n' \"$@\" > arguments\n");
            fixture.components(desktop);
            if desktop == Some(false) {
                let path = fixture.config().join(CONFIG_NAME);
                let mut config: serde_json::Value =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                config["backend"] = serde_json::json!("cpu");
                fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
            }
            let updater = SourceUpdater::default();
            updater.start(fixture.config()).unwrap();
            wait(&updater);
            assert_eq!(updater.status().state, "Completed");
            let arguments = fs::read_to_string(fixture.0.join("arguments")).unwrap();
            let backend = if cfg!(target_os = "macos") {
                "metal"
            } else {
                "cpu"
            };
            assert_eq!(
                arguments.lines().collect::<Vec<_>>(),
                vec![
                    "--update",
                    "--desktop",
                    "--backend",
                    backend,
                    "--jobs",
                    "2",
                    "--config-dir",
                    fixture.config().to_str().unwrap(),
                ]
            );
        }
    }

    #[cfg(unix)]
    fn wait(updater: &SourceUpdater) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while updater.is_running() && std::time::Instant::now() < deadline {
            thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!updater.is_running(), "fake update timed out");
    }
}
