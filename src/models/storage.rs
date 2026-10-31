//! Default lma-owned state paths. Resolving paths never creates or migrates files.
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn home_dir() -> Result<PathBuf, String> {
    let home = if cfg!(windows) {
        std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
    } else {
        std::env::var_os("HOME")
    };
    absolute_home(home.as_deref().map(Path::new))
}

fn absolute_home(home: Option<&Path>) -> Result<PathBuf, String> {
    home.filter(|path| path.is_absolute())
        .map(Path::to_path_buf)
        .ok_or_else(|| "cannot determine an absolute home directory; set HOME/USERPROFILE to an absolute path (for downloads, use `lma f MODEL -o DIR`)".into())
}

pub fn root() -> Result<PathBuf, String> {
    Ok(home_dir()?.join(".lma"))
}

fn environment_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

/// Former desktop config location, retained for read-only compatibility.
pub fn legacy_config_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        Some(
            home_dir()
                .ok()?
                .join("Library/Application Support/org.lma.app"),
        )
    } else if cfg!(windows) {
        Some(environment_dir("APPDATA")?.join("org.lma.app"))
    } else {
        Some(
            environment_dir("XDG_CONFIG_HOME")
                .or_else(|| home_dir().ok().map(|home| home.join(".config")))?
                .join("org.lma.app"),
        )
    }
}

/// Former desktop data location. Its weights remain in place.
pub fn legacy_data_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") || cfg!(windows) {
        legacy_config_dir()
    } else {
        Some(
            environment_dir("XDG_DATA_HOME")
                .or_else(|| home_dir().ok().map(|home| home.join(".local/share")))?
                .join("org.lma.app"),
        )
    }
}

/// Prefer the selected directory, including invalid files that need user repair.
/// Only the default directory may fall back; explicit host/test paths stay isolated.
/// Callers must still apply their usual regular-file, size and format validation.
pub fn read_path(directory: &Path, filename: &str) -> Result<PathBuf, String> {
    let legacy = if root().is_ok_and(|root| directory == root) {
        legacy_config_dir()
    } else {
        None
    };
    read_path_in(directory, legacy.as_deref(), filename)
}

fn read_path_in(
    directory: &Path,
    legacy: Option<&Path>,
    filename: &str,
) -> Result<PathBuf, String> {
    let primary = directory.join(filename);
    match fs::symlink_metadata(&primary) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(legacy.map_or(primary.clone(), |legacy| legacy.join(filename)))
        }
        Err(error) => Err(format!("cannot inspect {}: {error}", primary.display())),
        Ok(_) => Ok(primary),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_root_requires_absolute_home_without_creating_it() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        assert_eq!(
            absolute_home(Some(&home)).unwrap().join(".lma/models"),
            home.join(".lma/models")
        );
        assert!(!home.exists());
        for home in [None, Some(Path::new("")), Some(Path::new("relative"))] {
            assert!(absolute_home(home).is_err());
        }
    }

    #[test]
    fn legacy_reads_do_not_copy_and_primary_always_wins() {
        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join(".lma");
        let legacy = temp.path().join("old-config");
        fs::create_dir(&legacy).unwrap();
        let old = legacy.join("app-settings.json");
        fs::write(&old, b"legacy").unwrap();
        assert_eq!(
            read_path_in(&primary, Some(&legacy), "app-settings.json").unwrap(),
            old
        );
        assert!(!primary.exists());
        fs::create_dir(&primary).unwrap();
        let new = primary.join("app-settings.json");
        fs::write(&new, b"invalid primary must not be hidden").unwrap();
        assert_eq!(
            read_path_in(&primary, Some(&legacy), "app-settings.json").unwrap(),
            new
        );
        assert_eq!(fs::read(old).unwrap(), b"legacy");
    }

    #[test]
    fn explicit_directories_never_read_legacy_state() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("explicit");
        assert_eq!(
            read_path(&directory, "dictation.json").unwrap(),
            directory.join("dictation.json")
        );
        assert!(!directory.exists());
    }

    #[cfg(unix)]
    #[test]
    fn redirected_primary_is_not_hidden_by_legacy_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join(".lma");
        fs::create_dir(&primary).unwrap();
        let redirected = primary.join("app-settings.json");
        std::os::unix::fs::symlink(temp.path().join("missing"), &redirected).unwrap();
        assert_eq!(
            read_path_in(&primary, Some(temp.path()), "app-settings.json").unwrap(),
            redirected
        );
    }
}
