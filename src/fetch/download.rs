//! Bounded HTTP and streaming, verified, no-clobber publication.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::blocking::{Client, Response};
use reqwest::{StatusCode, Url};
use sha2::{Digest, Sha256};

use super::progress::{self, Event};

pub(super) const JSON_LIMIT: u64 = 8 * 1024 * 1024;
const BUFFER_SIZE: usize = 64 * 1024;

pub(super) fn client() -> Result<Client, String> {
    Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(15))
        // Bound blocking connect/read/write operations, including stalled body
        // reads. Catalog requests override this with 30s.
        .timeout(Duration::from_secs(60))
        // No ambient proxy credentials or automatically enabled decompression.
        .no_proxy()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() > 5 {
                attempt.error("more than five redirects")
            } else if !public_https(attempt.url()) {
                attempt.error("redirect must use HTTPS without URL credentials")
            } else {
                attempt.follow()
            }
        }))
        .referer(false)
        .user_agent("lma-fetch/0.1")
        .build()
        .map_err(|e| format!("cannot create HTTPS client: {e}"))
}

fn public_https(url: &Url) -> bool {
    url.scheme() == "https"
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
}

pub(super) fn checked_status(response: Response) -> Result<Response, String> {
    let status = response.status();
    if status.is_success() {
        Ok(response)
    } else if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
        Err(format!(
            "HTTP {status}: repository may be private/gated or the exact name may not exist (Hugging Face can return 401 for missing repositories); check owner/repository, or use -l to search. Authentication is not sent"
        ))
    } else {
        Err(format!("HTTP {status}"))
    }
}

// Never include signed redirect URLs in errors.
fn network_error(error: reqwest::Error) -> String {
    format!("HTTP request failed: {}", error.without_url())
}

pub(super) fn json(client: &Client, url: Url) -> Result<Option<serde_json::Value>, String> {
    let response = client
        .get(url)
        .header(
            reqwest::header::ACCEPT,
            "application/json, application/vnd.docker.distribution.manifest.v2+json",
        )
        .timeout(Duration::from_secs(30))
        .send()
        .map_err(network_error)?;
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let response = checked_status(response)?;
    let bytes = bounded_body(response, JSON_LIMIT)?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| format!("invalid catalog JSON: {e}"))
}

fn bounded_body(mut response: Response, limit: u64) -> Result<Vec<u8>, String> {
    if response.content_length().is_some_and(|size| size > limit) {
        return Err(format!("metadata exceeds {limit} bytes"));
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut response)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read metadata: {e}"))?;
    if bytes.len() as u64 > limit {
        return Err(format!("metadata exceeds {limit} bytes"));
    }
    Ok(bytes)
}

pub(super) fn safe_component(value: &str) -> bool {
    let stem = value.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
    !reserved
        && !value.is_empty()
        && value.len() <= 255
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && !value.ends_with('.')
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

pub(super) fn safe_relative(value: &str) -> Result<PathBuf, String> {
    if !value.split('/').all(safe_component) {
        return Err(format!(
            "unsafe relative filename {value:?}; percent-encoded, absolute, and traversal paths are not supported"
        ));
    }
    Ok(value.split('/').collect())
}

pub(super) fn sha256(value: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("expected a 64-hex SHA256 checksum; refusing unverified content".into());
    }
    Ok(value.to_ascii_lowercase())
}

pub(super) fn directory(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative = safe_relative(relative)?;
    fs::create_dir_all(root)
        .map_err(|e| format!("cannot create output directory '{}': {e}", root.display()))?;
    require_directory(root)?;
    let mut path = root.to_path_buf();
    for component in relative.components() {
        path.push(component);
        match fs::create_dir(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(format!("cannot create '{}': {e}", path.display())),
        }
        require_directory(&path)?;
    }
    Ok(path)
}

fn require_directory(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot inspect '{}': {e}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "'{}' must be a real directory, not a symlink",
            path.display()
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) struct Artifact {
    pub url: Url,
    pub filename: String,
    pub sha256: String,
    pub size: u64,
    pub gguf: bool,
}

pub(super) fn download(
    client: &Client,
    directory: &Path,
    artifact: &Artifact,
) -> Result<PathBuf, String> {
    download_with_progress(client, directory, artifact, &mut |_| {})
}

pub(super) fn download_cli(
    client: &Client,
    directory: &Path,
    artifact: &Artifact,
) -> Result<PathBuf, String> {
    download_with_progress(
        client,
        directory,
        artifact,
        &mut progress::stderr(&artifact.filename, artifact.size),
    )
}

fn download_with_progress(
    client: &Client,
    directory: &Path,
    artifact: &Artifact,
    progress: &mut impl FnMut(Event),
) -> Result<PathBuf, String> {
    let checksum = sha256(&artifact.sha256)?;
    let relative = safe_relative(&artifact.filename)?;
    let parent = match relative.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(parent) => self::directory(directory, &parent.to_string_lossy())?,
        None => {
            require_directory(directory)?;
            directory.to_path_buf()
        }
    };
    let destination = parent.join(relative.file_name().ok_or("missing filename")?);
    if existing(&destination, artifact.size, &checksum, artifact.gguf)? {
        println!("Verified existing file: {}", destination.display());
        return Ok(destination);
    }
    if !artifact.gguf && artifact.size > JSON_LIMIT {
        return Err("auxiliary metadata/license exceeds 8 MiB".into());
    }
    println!(
        "Downloading {} ({} bytes)",
        artifact.filename, artifact.size
    );
    progress(Event::Started);
    let result = (|| {
        let mut response = checked_status(
            client
                .get(artifact.url.clone())
                .send()
                .map_err(network_error)?,
        )?;
        if response
            .content_length()
            .is_some_and(|size| size != artifact.size)
        {
            return Err(format!(
                "Content-Length disagrees with trusted metadata for {}",
                artifact.filename
            ));
        }
        publish_with_progress(
            &mut response,
            &destination,
            artifact.size,
            &checksum,
            artifact.gguf,
            progress,
        )
    })();
    // Success is emitted only after checksum verification, sync and no-clobber publication.
    progress(Event::Finished(result.is_ok()));
    result?;
    println!(
        "Verified download: {} ({} bytes)",
        destination.display(),
        artifact.size
    );
    Ok(destination)
}

fn existing(path: &Path, size: u64, checksum: &str, gguf: bool) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(format!("cannot inspect '{}': {e}", path.display())),
        Ok(metadata) => {
            if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() != size {
                return Err(format!(
                    "existing '{}' differs or is not a regular file; move it aside explicitly before retrying",
                    path.display()
                ));
            }
        }
    }
    let mut file =
        File::open(path).map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
    verify_copy(&mut file, &mut std::io::sink(), size, checksum, gguf).map_err(|e| {
        format!(
            "existing '{}' is not valid: {e}; refusing to overwrite; move it aside explicitly",
            path.display()
        )
    })?;
    Ok(true)
}

fn publish(
    reader: &mut impl Read,
    path: &Path,
    size: u64,
    checksum: &str,
    gguf: bool,
) -> Result<(), String> {
    publish_with_progress(reader, path, size, checksum, gguf, &mut |_| {})
}

fn publish_with_progress(
    reader: &mut impl Read,
    path: &Path,
    size: u64,
    checksum: &str,
    gguf: bool,
    progress: &mut impl FnMut(Event),
) -> Result<(), String> {
    let mut temporary =
        tempfile::NamedTempFile::new_in(path.parent().ok_or("missing destination directory")?)
            .map_err(|e| format!("cannot reserve temporary file: {e}"))?;
    verify_copy_with_progress(reader, &mut temporary, size, checksum, gguf, progress)?;
    temporary
        .flush()
        .map_err(|e| format!("cannot flush download: {e}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|e| format!("cannot sync download: {e}"))?;
    temporary.persist_noclobber(path).map_err(|e| {
        format!(
            "cannot publish '{}' without overwriting: {}",
            path.display(),
            e.error
        )
    })?;
    Ok(())
}

fn verify_copy(
    reader: &mut impl Read,
    writer: &mut impl Write,
    size: u64,
    checksum: &str,
    gguf: bool,
) -> Result<(), String> {
    verify_copy_with_progress(reader, writer, size, checksum, gguf, &mut |_| {})
}

fn verify_copy_with_progress(
    reader: &mut impl Read,
    writer: &mut impl Write,
    size: u64,
    checksum: &str,
    gguf: bool,
    progress: &mut impl FnMut(Event),
) -> Result<(), String> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    let mut magic = [0_u8; 4];
    let mut count = 0_u64;
    let mut hash = Sha256::new();
    loop {
        let length = reader
            .read(&mut buffer)
            .map_err(|e| format!("cannot read content: {e}"))?;
        if length == 0 {
            break;
        }
        let next = count
            .checked_add(length as u64)
            .ok_or("content size overflow")?;
        if next > size {
            return Err("content exceeds size from trusted metadata".into());
        }
        for (index, byte) in buffer[..length]
            .iter()
            .take(4_u64.saturating_sub(count) as usize)
            .enumerate()
        {
            magic[count as usize + index] = *byte;
        }
        if gguf && next >= 4 && &magic != b"GGUF" {
            return Err(
                "content is not GGUF (invalid magic); adapters and other formats are unsupported"
                    .into(),
            );
        }
        hash.update(&buffer[..length]);
        writer
            .write_all(&buffer[..length])
            .map_err(|e| format!("cannot write content: {e}"))?;
        count = next;
        progress(Event::Bytes(count));
    }
    if count != size {
        return Err(format!(
            "size mismatch: expected {size} bytes, received {count}"
        ));
    }
    if gguf && (count < 4 || &magic != b"GGUF") {
        return Err("content is not GGUF (missing magic)".into());
    }
    if format!("{:x}", hash.finalize()) != checksum {
        return Err("SHA256 mismatch; content was not published".into());
    }
    Ok(())
}

pub(super) fn save_metadata(directory: &Path, filename: &str, bytes: &[u8]) -> Result<(), String> {
    if !safe_component(filename) || bytes.len() as u64 > JSON_LIMIT {
        return Err("invalid or oversized metadata snapshot".into());
    }
    require_directory(directory)?;
    let checksum = format!("{:x}", Sha256::digest(bytes));
    // Catalog counters and mutable tags can change between requests. Keep each
    // provenance snapshot rather than blocking a resume/projector fetch because
    // an unrelated metadata field changed, or overwriting its earlier origin.
    let stem = filename.strip_suffix(".json").unwrap_or(filename);
    let destination = directory.join(format!("{stem}-{checksum}.json"));
    if !existing(&destination, bytes.len() as u64, &checksum, false)? {
        publish(
            &mut &bytes[..],
            &destination,
            bytes.len() as u64,
            &checksum,
            false,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn mock(body: Vec<u8>, status: &str, extra: &str) -> (Client, Url, thread::JoinHandle<()>) {
        let headers = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n",
            body.len()
        );
        mock_raw(body, headers)
    }

    fn mock_raw(body: Vec<u8>, headers: String) -> (Client, Url, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = Url::parse(&format!("http://{}/test", listener.local_addr().unwrap())).unwrap();
        let task = thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "mock request never arrived"
                        );
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("{e}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            socket
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = [0; 4096];
            let _ = socket.read(&mut request);
            let _ = socket.write_all(headers.as_bytes());
            let _ = socket.write_all(&body);
        });
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        (client, url, task)
    }

    #[test]
    fn paths_and_digests_are_strict() {
        for path in [
            "../bad", "/bad", "a/../b", "a//b", "a\\b", "%2e%2e/x", "a/%252f", "a?b", "CON", "a.",
        ] {
            assert!(safe_relative(path).is_err(), "{path}");
        }
        assert!(safe_relative("owner/repo/sub/model.Q4_K_M.gguf").is_ok());
        assert!(sha256(&"A".repeat(64)).is_ok());
        assert!(sha256(&"z".repeat(64)).is_err());
        assert!(sha256("sha256:abcd").is_err());
        assert!(!public_https(
            &Url::parse("http://huggingface.co/a").unwrap()
        ));
        assert!(!public_https(
            &Url::parse("https://user:pass@huggingface.co/a").unwrap()
        ));
    }

    #[test]
    fn verified_atomic_no_clobber_and_cleanup() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let bytes = b"GGUFtiny model";
        let hash = format!("{:x}", Sha256::digest(bytes));
        publish(&mut &bytes[..], &path, bytes.len() as u64, &hash, true).unwrap();
        assert!(existing(&path, bytes.len() as u64, &hash, true).unwrap());
        assert!(publish(&mut &bytes[..], &path, bytes.len() as u64, &hash, true).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(existing(&path, 1, &hash, true).is_err());
        for (body, size, checksum, gguf) in [
            (&bytes[..], bytes.len() as u64, "0".repeat(64), true),
            (&bytes[..], 100, hash.clone(), true),
            (&bytes[..], 4, hash.clone(), true),
            (
                &b"NOPEtiny model"[..],
                bytes.len() as u64,
                hash.clone(),
                true,
            ),
        ] {
            assert!(
                publish(
                    &mut &body[..],
                    &directory.path().join("bad.gguf"),
                    size,
                    &checksum,
                    gguf
                )
                .is_err()
            );
            assert!(!directory.path().join("bad.gguf").exists());
        }
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn http_metadata_limits_status_and_download() {
        let (client, url, task) = mock(b"{}".to_vec(), "200 OK", "");
        assert_eq!(json(&client, url).unwrap(), Some(serde_json::json!({})));
        task.join().unwrap();
        let (client, url, task) = mock(vec![], "401 Unauthorized", "");
        assert!(json(&client, url).unwrap_err().contains("gated"));
        task.join().unwrap();
        let (client, url, task) = mock(vec![], "404 Not Found", "");
        assert!(json(&client, url).unwrap().is_none());
        task.join().unwrap();
        let (client, url, task) = mock(b"12345".to_vec(), "200 OK", "");
        assert!(bounded_body(client.get(url).send().unwrap(), 4).is_err());
        task.join().unwrap();
        let bytes = b"GGUFhttp model";
        let (client, url, task) = mock(bytes.to_vec(), "200 OK", "");
        let directory = tempfile::tempdir().unwrap();
        let artifact = Artifact {
            url,
            filename: "model.gguf".into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            size: bytes.len() as u64,
            gguf: true,
        };
        let path = download(&client, directory.path(), &artifact).unwrap();
        task.join().unwrap();
        assert_eq!(fs::read(path).unwrap(), bytes);
        // The server has stopped: this must skip without making another request.
        download(&client, directory.path(), &artifact).unwrap();
    }

    #[test]
    fn http_progress_counts_streamed_bytes_and_finishes_only_after_publication() {
        let mut bytes = vec![42; BUFFER_SIZE * 3 + 7];
        bytes[..4].copy_from_slice(b"GGUF");
        // No Content-Length: progress must use the trusted artifact total.
        let (client, url, task) = mock_raw(
            bytes.clone(),
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".into(),
        );
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("model.gguf");
        let artifact = Artifact {
            url,
            filename: "model.gguf".into(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            size: bytes.len() as u64,
            gguf: true,
        };
        let mut events = Vec::new();
        download_with_progress(&client, directory.path(), &artifact, &mut |event| {
            if event == Event::Finished(true) {
                assert_eq!(fs::read(&destination).unwrap(), bytes);
            } else {
                assert!(!destination.exists(), "no partial file may be published");
            }
            events.push(event);
        })
        .unwrap();
        task.join().unwrap();
        assert_eq!(events.first(), Some(&Event::Started));
        assert_eq!(events.last(), Some(&Event::Finished(true)));
        let counts: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Bytes(bytes) => Some(*bytes),
                _ => None,
            })
            .collect();
        assert!(counts.len() >= 4);
        assert!(counts.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(counts.last(), Some(&artifact.size));
        assert!(counts[0] <= BUFFER_SIZE as u64);
        // The mock server is gone; verified reuse emits no download events or requests.
        download_with_progress(&client, directory.path(), &artifact, &mut |_| {
            panic!("verified reuse must not report download progress")
        })
        .unwrap();
    }

    #[test]
    fn http_progress_errors_never_report_success_and_cleanup_temporary_files() {
        for (body, size, hash, headers) in [
            (
                b"GGUFbad".to_vec(),
                7,
                "0".repeat(64),
                "HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\n",
            ),
            (
                b"GGUFbad".to_vec(),
                8,
                "0".repeat(64),
                "HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\n",
            ),
            (
                b"GGUFbad".to_vec(),
                8,
                "0".repeat(64),
                "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            ),
            (
                b"GGUFbad".to_vec(),
                4,
                "0".repeat(64),
                "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            ),
            (
                vec![],
                7,
                "0".repeat(64),
                "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            ),
        ] {
            let (client, mut url, task) = mock_raw(body, headers.into());
            url.set_query(Some("token=never-display-this-secret"));
            let directory = tempfile::tempdir().unwrap();
            let artifact = Artifact {
                url,
                filename: "model.gguf".into(),
                sha256: hash,
                size,
                gguf: true,
            };
            let mut events = Vec::new();
            let error =
                download_with_progress(&client, directory.path(), &artifact, &mut |event| {
                    events.push(event);
                })
                .unwrap_err();
            task.join().unwrap();
            assert!(!error.contains("never-display-this-secret"));
            assert_eq!(events.first(), Some(&Event::Started));
            assert_eq!(events.last(), Some(&Event::Finished(false)));
            assert!(!events.contains(&Event::Finished(true)));
            assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
        }
    }

    #[test]
    fn progress_does_not_report_success_if_no_clobber_publication_loses_race() {
        let bytes = b"GGUFtiny model";
        let (client, url, task) = mock(bytes.to_vec(), "200 OK", "");
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("model.gguf");
        let artifact = Artifact {
            url,
            filename: "model.gguf".into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            size: bytes.len() as u64,
            gguf: true,
        };
        let mut events = Vec::new();
        let error = download_with_progress(&client, directory.path(), &artifact, &mut |event| {
            if event == Event::Bytes(artifact.size) {
                fs::write(&destination, b"do not replace").unwrap();
            }
            events.push(event);
        })
        .unwrap_err();
        task.join().unwrap();
        assert!(error.contains("without overwriting"));
        assert_eq!(events.last(), Some(&Event::Finished(false)));
        assert!(!events.contains(&Event::Finished(true)));
        assert_eq!(fs::read(&destination).unwrap(), b"do not replace");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn failed_http_downloads_leave_no_files() {
        for (body, status, size, hash, expected) in [
            (b"GGUFbad".to_vec(), "200 OK", 7, "0".repeat(64), "SHA256"),
            (b"NOTgguf".to_vec(), "200 OK", 7, "0".repeat(64), "magic"),
            (
                b"GGUFbad".to_vec(),
                "200 OK",
                8,
                "0".repeat(64),
                "Content-Length",
            ),
            (vec![], "403 Forbidden", 7, "0".repeat(64), "gated"),
            (vec![], "503 Unavailable", 7, "0".repeat(64), "503"),
        ] {
            let (client, url, task) = mock(body, status, "");
            let directory = tempfile::tempdir().unwrap();
            let artifact = Artifact {
                url,
                filename: "model.gguf".into(),
                sha256: hash,
                size,
                gguf: true,
            };
            let error = download(&client, directory.path(), &artifact).unwrap_err();
            assert!(error.contains(expected), "{error}");
            task.join().unwrap();
            assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
        }
    }

    #[test]
    fn metadata_without_content_length_is_still_bounded() {
        let (client, url, task) = mock_raw(
            b"12345".to_vec(),
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".into(),
        );
        assert!(
            bounded_body(client.get(url).send().unwrap(), 4)
                .unwrap_err()
                .contains("exceeds")
        );
        task.join().unwrap();
        let (client, url, task) = mock(b"not json".to_vec(), "200 OK", "");
        assert!(
            json(&client, url)
                .unwrap_err()
                .contains("invalid catalog JSON")
        );
        task.join().unwrap();
    }

    #[test]
    fn short_reads_io_failures_and_provenance_snapshots() {
        struct OneByte<'a>(&'a [u8]);
        impl Read for OneByte<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if self.0.is_empty() {
                    return Ok(0);
                }
                output[0] = self.0[0];
                self.0 = &self.0[1..];
                Ok(1)
            }
        }
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("broken connection"))
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let bytes = b"GGUFtiny";
        let checksum = format!("{:x}", Sha256::digest(bytes));
        publish(
            &mut OneByte(bytes),
            &directory.path().join("model.gguf"),
            bytes.len() as u64,
            &checksum,
            true,
        )
        .unwrap();
        assert!(
            publish(
                &mut Broken,
                &directory.path().join("bad.gguf"),
                10,
                &checksum,
                true
            )
            .unwrap_err()
            .contains("broken connection")
        );
        assert!(!directory.path().join("bad.gguf").exists());
        save_metadata(directory.path(), "hf-origin.json", b"{\"revision\":1}").unwrap();
        save_metadata(directory.path(), "hf-origin.json", b"{\"revision\":1}").unwrap();
        save_metadata(directory.path(), "hf-origin.json", b"{\"revision\":2}").unwrap();
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 3);
    }

    #[test]
    fn production_client_rejects_plain_http() {
        assert!(
            client()
                .unwrap()
                .get("http://127.0.0.1:9/no-request")
                .send()
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_output_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), root.path().join("escape")).unwrap();
        assert!(directory(root.path(), "escape/nested").is_err());
        std::os::unix::fs::symlink(
            elsewhere.path().join("missing"),
            root.path().join("model.gguf"),
        )
        .unwrap();
        assert!(existing(&root.path().join("model.gguf"), 0, &"0".repeat(64), true).is_err());
    }
}
