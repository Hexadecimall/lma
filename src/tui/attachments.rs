use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::{fs::File, io::Read, path::Path};

pub const MAX_IMAGES: usize = 8;
pub const IMAGE_LIMIT: usize = 8 * 1024 * 1024;
// The shared engine accepts at most 16 MiB decoded, inside a 24 MiB JSON request.
pub const TOTAL_IMAGE_LIMIT: usize = 16 * 1024 * 1024;
pub const REQUEST_LIMIT: usize = 24 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct Attachment {
    pub name: String,
    pub encoded: String,
    pub bytes: usize,
}

pub fn read(path: &Path, vision: bool, count: usize, total: usize) -> Result<Attachment, String> {
    if !vision {
        return Err("Images require a loaded vision projector (and a vision-enabled build). Choose one with /models or -pr FILE.".into());
    }
    if count >= MAX_IMAGES {
        return Err("At most 8 images may be present in a request, including history; use /new to clear it.".into());
    }
    // Reject devices/FIFOs before open, which itself could otherwise block.
    let metadata = std::fs::metadata(path).map_err(|e| format!("cannot inspect image: {e}"))?;
    if !metadata.is_file() {
        return Err("image must be a regular PNG or JPEG file".into());
    }
    if metadata.len() > IMAGE_LIMIT as u64 {
        return Err("image exceeds 8 MiB".into());
    }
    let file = File::open(path).map_err(|e| format!("cannot open image: {e}"))?;
    if !file.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("image must be a regular PNG or JPEG file".into());
    }
    let mut bytes = Vec::new();
    file.take((IMAGE_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read image: {e}"))?;
    if bytes.len() > IMAGE_LIMIT {
        return Err("image exceeds 8 MiB".into());
    }
    if total.saturating_add(bytes.len()) > TOTAL_IMAGE_LIMIT {
        return Err("Images in this request exceed the engine's 16 MiB decoded limit; use /new to clear history.".into());
    }
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") && !bytes.starts_with(b"\xff\xd8\xff") {
        return Err(
            "attachment must be PNG or JPEG; the engine also validates its decoded contents".into(),
        );
    }
    Ok(Attachment {
        name: path.to_string_lossy().into_owned(),
        encoded: STANDARD.encode(&bytes),
        bytes: bytes.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_signatures_and_enforces_byte_budgets() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"not an image").unwrap();
        assert!(
            read(file.path(), true, 0, 0)
                .unwrap_err()
                .contains("PNG or JPEG")
        );
        file.as_file().set_len(0).unwrap();
        file.reopen()
            .unwrap()
            .write_all(b"\x89PNG\r\n\x1a\n")
            .unwrap();
        assert_eq!(read(file.path(), true, 0, 0).unwrap().bytes, 8);
        assert!(
            read(file.path(), true, 0, TOTAL_IMAGE_LIMIT)
                .unwrap_err()
                .contains("16 MiB")
        );
        file.as_file().set_len((IMAGE_LIMIT + 1) as u64).unwrap();
        assert!(read(file.path(), true, 0, 0).unwrap_err().contains("8 MiB"));
    }
    #[test]
    fn checks_projector_and_count_before_reading() {
        assert!(
            read(Path::new("missing.png"), false, 0, 0)
                .unwrap_err()
                .contains("projector")
        );
        assert!(
            read(Path::new("missing.png"), true, 8, 0)
                .unwrap_err()
                .contains("8 images")
        );
    }
}
