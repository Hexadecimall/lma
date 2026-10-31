//! Open local assets without blocking on special files. Parent directories are trusted.
use std::{
    fs::{self, File, OpenOptions},
    path::Path,
};

pub(crate) fn open_regular(path: &Path) -> Result<File, String> {
    let before =
        fs::metadata(path).map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
    if !before.is_file() {
        return Err(format!("{} must be a regular file", path.display()));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    // Unix permits ordinary symlinked assets, without blocking on a raced FIFO.
    // Windows rejects reparse points rather than following redirected assets.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x800); // O_NONBLOCK
    }
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x4); // O_NONBLOCK
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options
        .open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if !after.is_file() || after.file_type().is_symlink() {
        return Err(format!("{} must be a regular file", path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Err(format!("{} changed while opening it", path.display()));
        }
    }
    Ok(file)
}
