//! Local, single-administrator OpenAI Files storage. No inference or network access.
//!
//! Metadata is the commit marker: a file is visible only after both private data
//! and metadata have been published. All readers and mutations take the same OS
//! lock. The storage directory must not be modified by other programs; an actor
//! running as this OS user is outside the single-admin security boundary.
use super::server::ApiError;
use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request, State},
    http::{Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io::{Read, Write},
    path::{Component, Path as FsPath, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tempfile::NamedTempFile;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_stream::StreamExt;

const MAX_FILE: u64 = 512 * 1024 * 1024;
const MAX_STORE: u64 = 2 * 1024 * 1024 * 1024;
const MAX_FILES: usize = 10_000;
const MAX_METADATA: u64 = 4096;
const MAX_FILENAME: usize = 255;
const CHUNK: usize = 64 * 1024;
const BODY_LIMIT: usize = MAX_FILE as usize + 64 * 1024;
const MAX_PART_HEADERS: usize = 16 * 1024;
const MAX_PREAMBLE: usize = 4096;
const UPLOAD_IDLE: Duration = Duration::from_secs(30);
const UPLOAD_DEADLINE: Duration = Duration::from_secs(600);
const UPLOAD_PURPOSES: &[&str] = &[
    "assistants",
    "batch",
    "fine-tune",
    "vision",
    "user_data",
    "evals",
];

fn internal() -> ApiError {
    ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "file_storage_error",
        "private file storage is unavailable or contains unsafe or invalid entries",
    )
}
fn recovery_required() -> ApiError {
    ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "file_storage_recovery_required",
        "file storage has an incomplete operation; recovery records were preserved in the private Files directory; resolve conflicting or unsafe entries before retrying",
    )
}
fn missing() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "file_not_found", "file not found")
}
fn quota() -> ApiError {
    ApiError::invalid("file storage quota exceeded (2 GiB and 10,000 files maximum)")
}
fn too_large() -> ApiError {
    ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "file_too_large",
        "file exceeds 512 MiB or multipart body exceeds its bounded overhead allowance",
    )
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|_| internal())?
}

/// Persisted/API metadata. `status` is always `uploaded`, not a processing claim.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileMetadata {
    pub(super) id: String,
    pub(super) object: String,
    pub(super) bytes: u64,
    pub(super) created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expires_at: Option<u64>,
    pub(super) filename: String,
    pub(super) purpose: String,
    pub(super) status: String,
}
impl FileMetadata {
    fn expired(&self) -> bool {
        self.expires_at.is_some_and(|expiry| expiry <= now())
    }
}

/// Private disk store for Files and future Batches. Construction does no I/O.
/// All methods use the single-admin scope; IDs are not authorization credentials.
#[derive(Clone)]
pub(super) struct FileStore {
    root: PathBuf,
    max_file: u64,
    max_store: u64,
    max_files: usize,
}

/// Validated listing options. Stable `(created_at, id)` ordering breaks timestamp ties.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ListOptions {
    pub(super) purpose: Option<String>,
    pub(super) after: Option<String>,
    #[serde(default = "default_limit")]
    pub(super) limit: usize,
    #[serde(default = "default_order")]
    pub(super) order: String,
}
fn default_limit() -> usize {
    10_000
}
fn default_order() -> String {
    "desc".into()
}
impl Default for ListOptions {
    fn default() -> Self {
        Self {
            purpose: None,
            after: None,
            limit: default_limit(),
            order: default_order(),
        }
    }
}

impl FileStore {
    /// Resolve `~/.lma/openai/files` without creating it or consulting legacy paths.
    pub(super) fn open_default() -> Result<Self, ApiError> {
        Self::default_location().map(Self::new)
    }

    fn default_location() -> Result<PathBuf, ApiError> {
        let models = crate::models::discovery::default_models_dir().map_err(|_| internal())?;
        Ok(models.parent().ok_or_else(internal)?.join("openai/files"))
    }

    /// Supply an absolute, non-symlink root (e.g. a canonical temporary directory).
    /// Missing directories are created privately only by the first mutation.
    pub(super) fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_file: MAX_FILE,
            max_store: MAX_STORE,
            max_files: MAX_FILES,
        }
    }

    /// Get unexpired metadata without reading the uploaded bytes.
    pub(super) async fn metadata(&self, id: &str) -> Result<FileMetadata, ApiError> {
        let store = self.clone();
        let id = id.to_owned();
        blocking(move || {
            let _lock = store.lock(false)?.ok_or_else(missing)?;
            store.metadata_locked(&id, false)
        })
        .await
    }

    /// Open an unexpired regular data file, checking its size against metadata.
    /// The returned handle remains usable if a concurrent delete unlinks the file.
    /// Consumers must stream it or enforce their own bounded read (see `read`).
    pub(super) async fn open_content(
        &self,
        id: &str,
    ) -> Result<(FileMetadata, tokio::fs::File), ApiError> {
        let store = self.clone();
        let id = id.to_owned();
        let (metadata, file) = blocking(move || {
            let _lock = store.lock(false)?.ok_or_else(missing)?;
            let metadata = store.metadata_locked(&id, false)?;
            let file = store.data_locked(&metadata)?;
            Ok((metadata, file))
        })
        .await?;
        Ok((metadata, tokio::fs::File::from_std(file)))
    }

    /// Read for a bounded consumer, e.g. a Batch input parser. Refuses files above
    /// `max_bytes` before allocating; the absolute 512 MiB ceiling always applies.
    pub(super) async fn read(&self, id: &str, max_bytes: u64) -> Result<Vec<u8>, ApiError> {
        let (metadata, file) = self.open_content(id).await?;
        if metadata.bytes > max_bytes.min(self.max_file) {
            return Err(ApiError::invalid("file exceeds the consumer's read limit"));
        }
        let mut bytes = Vec::new();
        file.take(metadata.bytes + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| internal())?;
        if bytes.len() as u64 != metadata.bytes {
            return Err(internal());
        }
        Ok(bytes)
    }

    /// Publish bounded in-memory output (e.g. Batch output/error JSONL). Does not
    /// clone the slice. Generated output purposes are accepted only by this core
    /// method, never by multipart uploads. Use streaming upload for large input.
    pub(super) async fn store_bytes(
        &self,
        filename: &str,
        purpose: &str,
        bytes: &[u8],
        expires_after: Option<u64>,
    ) -> Result<FileMetadata, ApiError> {
        validate_filename(filename)?;
        validate_purpose(purpose, true)?;
        validate_expiry(expires_after)?;
        if bytes.len() as u64 > self.max_file {
            return Err(too_large());
        }
        let mut pending = self.begin().await?;
        pending.check_size(bytes.len() as u64)?;
        let mut file =
            tokio::fs::File::from_std(pending.temp.as_file().try_clone().map_err(|_| internal())?);
        for chunk in bytes.chunks(CHUNK) {
            file.write_all(chunk).await.map_err(|_| internal())?;
        }
        file.sync_all().await.map_err(|_| internal())?;
        drop(file);
        pending.bytes = bytes.len() as u64;
        self.finish(pending, filename.into(), purpose.into(), expires_after)
            .await
    }

    /// List active files; missing storage is an empty list and creates nothing.
    /// `after` must identify an active file, even when a purpose filter is used.
    pub(super) async fn list(&self, options: ListOptions) -> Result<Value, ApiError> {
        if !(1..=10_000).contains(&options.limit)
            || !matches!(options.order.as_str(), "asc" | "desc")
        {
            return Err(ApiError::invalid(
                "limit must be 1..10000 and order asc or desc",
            ));
        }
        if let Some(id) = &options.after {
            validate_id(id)?;
        }
        if options.purpose.as_ref().is_some_and(|p| p.len() > 64) {
            return Err(ApiError::invalid("purpose filter is too long"));
        }
        let store = self.clone();
        blocking(move || {
            let Some(_lock) = store.lock(false)? else {
                return if options.after.is_some() {
                    Err(missing())
                } else {
                    Ok(list_value(&[], false))
                };
            };
            let mut files = store.scan_locked()?;
            files.retain(|file| !file.expired());
            files.sort_by(|a, b| (a.created_at, &a.id).cmp(&(b.created_at, &b.id)));
            if options.order == "desc" {
                files.reverse();
            }
            if let Some(after) = &options.after {
                let index = files
                    .iter()
                    .position(|file| &file.id == after)
                    .ok_or_else(missing)?;
                files.drain(..=index);
            }
            if let Some(purpose) = &options.purpose {
                files.retain(|file| &file.purpose == purpose);
            }
            let has_more = files.len() > options.limit;
            files.truncate(options.limit);
            Ok(list_value(&files, has_more))
        })
        .await
    }

    /// Delete only this store's validated metadata/data pair. Existing open read
    /// handles may finish streaming; no external references or vector stores exist.
    pub(super) async fn delete(&self, id: &str) -> Result<(), ApiError> {
        let store = self.clone();
        let id = id.to_owned();
        blocking(move || {
            let _lock = store.lock(false)?.ok_or_else(missing)?;
            store.clean_staging_locked()?;
            let metadata = store.metadata_locked(&id, false)?;
            store.remove_locked(&metadata)
        })
        .await
    }

    async fn begin(&self) -> Result<Pending, ApiError> {
        let store = self.clone();
        blocking(move || {
            let lock = store.lock(true)?.ok_or_else(internal)?;
            store.clean_staging_locked()?;
            let files = store.scan_locked()?;
            let mut used = 0u64;
            let mut count = 0usize;
            for metadata in files {
                if metadata.expired() {
                    store.remove_locked(&metadata)?;
                } else {
                    used = used.checked_add(metadata.bytes).ok_or_else(quota)?;
                    count += 1;
                }
            }
            if count >= store.max_files || used >= store.max_store {
                return Err(quota());
            }
            let temp = tempfile::Builder::new()
                .prefix(".upload-")
                .rand_bytes(24)
                .tempfile_in(&store.root)
                .map_err(|_| internal())?;
            Ok(Pending {
                temp,
                _lock: lock,
                bytes: 0,
                used,
                max_file: store.max_file,
                max_store: store.max_store,
            })
        })
        .await
    }

    async fn finish(
        &self,
        pending: Pending,
        filename: String,
        purpose: String,
        expires_after: Option<u64>,
    ) -> Result<FileMetadata, ApiError> {
        let store = self.clone();
        blocking(move || store.publish_locked(pending, filename, purpose, expires_after)).await
    }

    fn publish_locked(
        &self,
        pending: Pending,
        filename: String,
        purpose: String,
        expires_after: Option<u64>,
    ) -> Result<FileMetadata, ApiError> {
        pending.check_size(pending.bytes)?;
        validate_filename(&filename)?;
        validate_purpose(&purpose, true)?;
        validate_expiry(expires_after)?;
        let suffix = pending
            .temp
            .path()
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix(".upload-"))
            .ok_or_else(internal)?;
        let id = format!("file-{suffix}");
        validate_id(&id)?;
        let created_at = now();
        let expiry = expires_after.or_else(|| (purpose == "batch").then_some(2_592_000));
        let metadata = FileMetadata {
            id: id.clone(),
            object: "file".into(),
            bytes: pending.bytes,
            created_at,
            expires_at: expiry.map(|seconds| created_at + seconds),
            filename,
            purpose,
            status: "uploaded".into(),
        };
        let encoded = serde_json::to_vec(&metadata).map_err(|_| internal())?;
        if encoded.len() as u64 > MAX_METADATA {
            return Err(internal());
        }
        let mut meta = tempfile::Builder::new()
            .prefix(".metadata-")
            .tempfile_in(&self.root)
            .map_err(|_| internal())?;
        meta.write_all(&encoded).map_err(|_| internal())?;
        meta.as_file().sync_all().map_err(|_| internal())?;
        pending.temp.as_file().sync_all().map_err(|_| internal())?;
        let data_path = self.root.join(format!("{id}.data"));
        let metadata_path = self.root.join(format!("{id}.json"));
        // Preserve the journal before publishing bytes: an interrupted writer must
        // leave enough information to recover a data file without its commit marker.
        meta.disable_cleanup(true);
        let _data = match publish_file(pending.temp, &data_path) {
            Ok(file) => file,
            Err(_) => {
                // The pre-existing path is not ours; never recover it using this
                // upload's metadata on a later request.
                meta.disable_cleanup(false);
                return Err(internal());
            }
        };
        if let Err(error) = publish_file(meta, &metadata_path) {
            return Err(Self::rollback_publication(error.file, &data_path, |path| {
                fs::remove_file(path)
            }));
        }
        Ok(metadata)
    }

    fn rollback_publication(
        mut marker: NamedTempFile,
        data_path: &FsPath,
        unlink: impl FnOnce(&FsPath) -> std::io::Result<()>,
    ) -> ApiError {
        if unlink(data_path).is_err() {
            marker.disable_cleanup(true);
            recovery_required()
        } else {
            marker.disable_cleanup(false);
            internal()
        }
    }

    fn read_metadata(&self, path: &FsPath) -> Result<FileMetadata, ApiError> {
        let file = open_regular(path, false)?;
        let size = file.metadata().map_err(|_| internal())?.len();
        if size > MAX_METADATA {
            return Err(internal());
        }
        let metadata: FileMetadata =
            serde_json::from_reader(file.take(MAX_METADATA + 1)).map_err(|_| internal())?;
        validate_id(&metadata.id).map_err(|_| internal())?;
        if metadata.object != "file"
            || metadata.status != "uploaded"
            || metadata.bytes > self.max_file
            || metadata.created_at > now().saturating_add(60)
        {
            return Err(internal());
        }
        validate_filename(&metadata.filename).map_err(|_| internal())?;
        validate_purpose(&metadata.purpose, true).map_err(|_| internal())?;
        Ok(metadata)
    }

    fn metadata_locked(&self, id: &str, include_expired: bool) -> Result<FileMetadata, ApiError> {
        validate_id(id)?;
        let metadata = self.read_metadata(&self.root.join(format!("{id}.json")))?;
        if metadata.id != id {
            return Err(internal());
        }
        if !include_expired && metadata.expired() {
            return Err(missing());
        }
        // Broken/incomplete pairs must never appear as successfully stored files.
        self.data_locked(&metadata)?;
        Ok(metadata)
    }

    fn data_locked(&self, metadata: &FileMetadata) -> Result<File, ApiError> {
        let file = open_regular(&self.root.join(format!("{}.data", metadata.id)), false)?;
        if file.metadata().map_err(|_| internal())?.len() != metadata.bytes {
            return Err(internal());
        }
        Ok(file)
    }

    fn scan_locked(&self) -> Result<Vec<FileMetadata>, ApiError> {
        let mut files = Vec::new();
        let mut entries = 0usize;
        for entry in fs::read_dir(&self.root).map_err(|_| internal())? {
            entries += 1;
            // Bound scan cost even if a crashed writer left staging/orphan files.
            if entries > self.max_files.saturating_mul(3).saturating_add(32) {
                return Err(quota());
            }
            let entry = entry.map_err(|_| internal())?;
            let name = entry.file_name();
            let name = name.to_str().ok_or_else(internal)?;
            if let Some(id) = name.strip_suffix(".json") {
                if !id.starts_with("file-") {
                    continue;
                }
                if files.len() >= self.max_files {
                    return Err(quota());
                }
                files.push(self.metadata_locked(id, true)?);
            } else if name.starts_with("file-") && name.ends_with(".data") {
                // Account for orphan data conservatively: never silently reuse its quota.
                let id = name.strip_suffix(".data").ok_or_else(internal)?;
                validate_id(id).map_err(|_| internal())?;
                open_regular(&entry.path(), false)?;
                match fs::symlink_metadata(self.root.join(format!("{id}.json"))) {
                    Ok(_) => {}
                    Err(_) => return Err(internal()),
                }
            }
        }
        Ok(files)
    }

    fn clean_staging_locked(&self) -> Result<(), ApiError> {
        // Recover metadata before reclaiming abandoned uploads. The OS lock proves
        // no cooperating writer is still using any of these staging records.
        let mut uploads = Vec::new();
        for (index, entry) in fs::read_dir(&self.root)
            .map_err(|_| internal())?
            .enumerate()
        {
            if index >= self.max_files.saturating_mul(3).saturating_add(32) {
                return Err(quota());
            }
            let entry = entry.map_err(|_| internal())?;
            let name = entry.file_name();
            let name = name.to_str().ok_or_else(internal)?;
            if name.starts_with(".metadata-") || name.starts_with(".deleted-") {
                let path = entry.path();
                let file = open_regular(&path, false).map_err(|_| recovery_required())?;
                if file.metadata().map_err(|_| recovery_required())?.len() == 0 {
                    // A crash before writing/withdrawing metadata cannot own data.
                    fs::remove_file(path).map_err(|_| recovery_required())?;
                    continue;
                }
                let metadata = self.read_metadata(&path).map_err(|_| recovery_required())?;
                let data_path = self.root.join(format!("{}.data", metadata.id));
                match fs::symlink_metadata(&data_path) {
                    Ok(_) => {
                        self.data_locked(&metadata)
                            .map_err(|_| recovery_required())?;
                        self.restore_marker_locked(&path, &metadata.id)?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        // No published bytes: either publication never started or
                        // deletion finished. Only the journal is now redundant.
                        fs::remove_file(path).map_err(|_| recovery_required())?;
                    }
                    Err(_) => return Err(recovery_required()),
                }
            } else if name.starts_with(".upload-") {
                uploads.push(entry.path());
            }
        }
        for path in uploads {
            open_regular(&path, false)?;
            fs::remove_file(path).map_err(|_| internal())?;
        }
        Ok(())
    }

    fn restore_marker_locked(&self, marker: &FsPath, id: &str) -> Result<(), ApiError> {
        let target = self.root.join(format!("{id}.json"));
        match fs::symlink_metadata(&target) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // All cooperating writers hold the same lock. A conflicting
                // record is never replaced, even when it has matching contents.
                rename_noreplace(marker, &target).map_err(|_| recovery_required())
            }
            _ => Err(recovery_required()),
        }
    }

    fn remove_locked(&self, metadata: &FileMetadata) -> Result<(), ApiError> {
        self.data_locked(metadata)?;
        let path = self.root.join(format!("{}.json", metadata.id));
        open_regular(&path, false)?;
        let mut tombstone = tempfile::Builder::new()
            .prefix(".deleted-")
            .tempfile_in(&self.root)
            .map_err(|_| internal())?;
        // Withdraw the commit marker first; on data-unlink failure restore it.
        // Readers share the lock and cannot observe this intermediate state.
        fs::rename(&path, tombstone.path()).map_err(|_| internal())?;
        tombstone.disable_cleanup(true);
        if fs::remove_file(self.root.join(format!("{}.data", metadata.id))).is_err() {
            self.restore_marker_locked(tombstone.path(), &metadata.id)?;
            return Err(internal());
        }
        // Logical deletion succeeded. If unlinking this journal fails, the next
        // mutation recognizes its missing data and safely finishes cleanup.
        tombstone.disable_cleanup(false);
        drop(tombstone);
        Ok(())
    }

    fn lock(&self, create: bool) -> Result<Option<File>, ApiError> {
        if !self.ensure_root(create)? {
            return Ok(None);
        }
        let path = self.root.join(".files.lock");
        let mut options = safe_options(true);
        let file = match options.create_new(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                open_regular(&path, true)?
            }
            Err(_) => return Err(internal()),
        };
        check_regular(&file.metadata().map_err(|_| internal())?)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(file)),
            Err(TryLockError::WouldBlock) => Err(ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "file_store_busy",
                "file storage is busy; retry the request",
            )),
            Err(TryLockError::Error(_)) => Err(internal()),
        }
    }

    fn ensure_root(&self, create: bool) -> Result<bool, ApiError> {
        // std cannot establish private Windows ACLs; fail closed on unsupported
        // platforms rather than expose private code uploads with inherited ACLs.
        if !cfg!(any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos"
        )) {
            return Err(internal());
        }
        if !self.root.is_absolute() {
            return Err(internal());
        }
        let mut path = PathBuf::new();
        for component in self.root.components() {
            match component {
                Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                    path.push(component.as_os_str())
                }
                _ => return Err(internal()),
            }
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound && !create => {
                    return Ok(false);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let mut builder = fs::DirBuilder::new();
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::DirBuilderExt;
                        builder.mode(0o700);
                    }
                    match builder.create(&path) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                        Err(_) => return Err(internal()),
                    }
                    fs::symlink_metadata(&path).map_err(|_| internal())?
                }
                Err(_) => return Err(internal()),
            };
            if !metadata.is_dir() || redirected(&metadata) {
                return Err(internal());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                // A non-sticky writable ancestor would let another OS user replace
                // our private directory despite its own restrictive permissions.
                if metadata.mode() & 0o022 != 0 && metadata.mode() & 0o1000 == 0 {
                    return Err(internal());
                }
            }
            if path == self.root {
                check_private(&metadata)?;
            }
        }
        Ok(true)
    }
}

struct Pending {
    temp: NamedTempFile,
    _lock: File,
    bytes: u64,
    used: u64,
    max_file: u64,
    max_store: u64,
}
impl Pending {
    fn check_size(&self, bytes: u64) -> Result<(), ApiError> {
        if bytes > self.max_file {
            return Err(too_large());
        }
        if self
            .used
            .checked_add(bytes)
            .is_none_or(|size| size > self.max_store)
        {
            return Err(quota());
        }
        Ok(())
    }
}
fn list_value(files: &[FileMetadata], has_more: bool) -> Value {
    json!({"object":"list", "data":files, "first_id":files.first().map(|f| &f.id),
        "last_id":files.last().map(|f| &f.id), "has_more":has_more})
}
fn validate_id(id: &str) -> Result<(), ApiError> {
    let valid = id.strip_prefix("file-").is_some_and(|suffix| {
        (6..=64).contains(&suffix.len()) && suffix.bytes().all(|c| c.is_ascii_alphanumeric())
    });
    if valid { Ok(()) } else { Err(missing()) }
}
fn validate_filename(filename: &str) -> Result<(), ApiError> {
    if filename.is_empty()
        || filename.len() > MAX_FILENAME
        || filename.chars().any(char::is_control)
    {
        return Err(ApiError::invalid(
            "file must have a nonempty filename of at most 255 UTF-8 bytes without control characters",
        ));
    }
    // The filename is display metadata, never a filesystem path.
    Ok(())
}
fn validate_purpose(purpose: &str, generated: bool) -> Result<(), ApiError> {
    if UPLOAD_PURPOSES.contains(&purpose)
        || (generated
            && ["batch_output", "assistants_output", "fine-tune-results"].contains(&purpose))
    {
        Ok(())
    } else {
        Err(ApiError::invalid(
            "purpose must be assistants, batch, fine-tune, vision, user_data, or evals",
        ))
    }
}
fn validate_expiry(expiry: Option<u64>) -> Result<(), ApiError> {
    if expiry.is_some_and(|seconds| !(3600..=2_592_000).contains(&seconds)) {
        Err(ApiError::invalid(
            "expires_after.seconds must be between 3600 and 2592000",
        ))
    } else {
        Ok(())
    }
}
fn redirected(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return true;
        }
    }
    metadata.file_type().is_symlink()
}
fn check_private(metadata: &fs::Metadata) -> Result<(), ApiError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        unsafe extern "C" {
            fn geteuid() -> u32;
        }
        // SAFETY: geteuid has no arguments and no memory-safety preconditions.
        if metadata.uid() != unsafe { geteuid() } || metadata.mode() & 0o077 != 0 {
            return Err(internal());
        }
    }
    Ok(())
}
fn check_regular(metadata: &fs::Metadata) -> Result<(), ApiError> {
    if !metadata.is_file() || redirected(metadata) {
        return Err(internal());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() != 1 {
            return Err(internal());
        }
    }
    check_private(metadata)
}
fn safe_options(write: bool) -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true).write(write);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    options
}
// Never use tempfile's hard-link fallback: an interrupted link/unlink pair
// violates this store's single-link invariant. Unsupported filesystems fail
// closed instead of weakening publication or silently overwriting a destination.
fn rename_noreplace(source: &FsPath, target: &FsPath) -> std::io::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        use rustix::fs::{CWD, RenameFlags, renameat_with};
        renameat_with(CWD, source, CWD, target, RenameFlags::NOREPLACE)
            .map_err(std::io::Error::from)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = (source, target);
        Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
    }
}

fn publish_file(temp: NamedTempFile, target: &FsPath) -> Result<File, tempfile::PersistError> {
    match rename_noreplace(temp.path(), target) {
        Ok(()) => Ok(temp.into_file()),
        Err(error) => Err(tempfile::PersistError { error, file: temp }),
    }
}

fn open_regular(path: &FsPath, write: bool) -> Result<File, ApiError> {
    let before = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            missing()
        } else {
            internal()
        }
    })?;
    check_regular(&before)?;
    let file = safe_options(write).open(path).map_err(|_| internal())?;
    let opened = file.metadata().map_err(|_| internal())?;
    check_regular(&opened)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != opened.dev() || before.ino() != opened.ino() {
            return Err(internal());
        }
    }
    Ok(file)
}

#[derive(Clone)]
struct FileState {
    store: Option<FileStore>,
    api_key: Option<String>,
    loopback: bool,
}
impl FileState {
    fn store(&self) -> Result<&FileStore, ApiError> {
        self.store.as_ref().ok_or_else(internal)
    }
}
/// Independent state/auth router, suitable for merging into a hosting Router<S>.
/// The caller should expose it only for the `all` and `openai` route profiles.
pub(super) fn router<S>(config: &crate::config::ServeConfig) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let loopback = config.host == "localhost"
        || config
            .host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    file_router(FileState {
        store: FileStore::open_default().ok(),
        api_key: config.api_key.clone(),
        loopback,
    })
}
fn file_router<S: Clone + Send + Sync + 'static>(state: FileState) -> Router<S> {
    Router::new()
        .route("/v1/files", get(list_files).post(upload))
        .route("/v1/files/{file_id}", get(metadata).delete(delete_file))
        .route("/v1/files/{file_id}/content", get(content))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(middleware::from_fn_with_state(state.clone(), authorize))
        .with_state(state)
}
async fn authorize(State(state): State<FileState>, request: Request, next: Next) -> Response {
    let head = request.method() == Method::HEAD;
    let error = if let Some(key) = &state.api_key {
        let authorized = !key.is_empty()
            && !key.chars().any(char::is_whitespace)
            && request
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|h| h.to_str().ok())
                .and_then(|h| h.split_once(' '))
                .is_some_and(|(scheme, token)| {
                    scheme.eq_ignore_ascii_case("bearer")
                        && constant_time_equal(token.as_bytes(), key.as_bytes())
                });
        (!authorized).then(|| {
            ApiError::new(
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "a valid Authorization: Bearer <api-key> header is required",
            )
        })
    } else if !state.loopback {
        Some(ApiError::new(
            StatusCode::FORBIDDEN,
            "admin_auth_required",
            "file storage on a non-loopback host requires an API key",
        ))
    } else {
        None
    };
    let mut response = if let Some(error) = error {
        let mut response = error.into_response();
        if response.status() == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, "Bearer".parse().unwrap());
        }
        response
    } else {
        next.run(request).await
    };
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    if head {
        *response.body_mut() = Body::empty();
    }
    response
}
fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    // Compare fixed-size hashes so unequal token lengths also take the same
    // comparison path. Hashing cost depends on public header length, not key data.
    use sha2::{Digest, Sha256};
    let left = Sha256::digest(left);
    let right = Sha256::digest(right);
    let mut difference = 0u8;
    for (a, b) in left.iter().zip(right.iter()) {
        difference |= a ^ b;
    }
    std::hint::black_box(difference) == 0
}
async fn list_files(
    State(state): State<FileState>,
    query: Result<Query<ListOptions>, axum::extract::rejection::QueryRejection>,
) -> Result<Json<Value>, ApiError> {
    let Query(options) =
        query.map_err(|_| ApiError::invalid("invalid or duplicate file listing parameters"))?;
    Ok(Json(state.store()?.list(options).await?))
}
async fn metadata(
    State(state): State<FileState>,
    path: Result<Path<String>, axum::extract::rejection::PathRejection>,
) -> Result<Json<FileMetadata>, ApiError> {
    let Path(id) = path.map_err(|_| missing())?;
    Ok(Json(state.store()?.metadata(&id).await?))
}
async fn delete_file(
    State(state): State<FileState>,
    path: Result<Path<String>, axum::extract::rejection::PathRejection>,
) -> Result<Json<Value>, ApiError> {
    let Path(id) = path.map_err(|_| missing())?;
    state.store()?.delete(&id).await?;
    Ok(Json(json!({"id":id, "object":"file", "deleted":true})))
}
async fn content(
    State(state): State<FileState>,
    method: Method,
    path: Result<Path<String>, axum::extract::rejection::PathRejection>,
) -> Result<Response, ApiError> {
    let Path(id) = path.map_err(|_| missing())?;
    let (metadata, mut file) = state.store()?.open_content(&id).await?;
    let body = if method == Method::HEAD {
        Body::empty()
    } else {
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(2);
        tokio::spawn(async move {
            let mut remaining = metadata.bytes;
            while remaining > 0 {
                let mut buffer = vec![0; remaining.min(CHUNK as u64) as usize];
                match file.read(&mut buffer).await {
                    Ok(0) => {
                        let _ = sender
                            .send(Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)))
                            .await;
                        break;
                    }
                    Ok(length) => {
                        buffer.truncate(length);
                        remaining -= length as u64;
                        if sender.send(Ok(buffer)).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        let _ = sender
                            .send(Err(std::io::Error::other("file read failed")))
                            .await;
                        break;
                    }
                }
            }
        });
        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(receiver))
    };
    // Suggest only a basename, even though metadata can retain a client path.
    // ASCII fallback plus percent-encoded UTF-8 RFC 5987 value; never interpolate
    // raw user-supplied filename bytes into an HTTP header.
    let filename = metadata
        .filename
        .rsplit(['/', '\\'])
        .next()
        .filter(|name| !name.is_empty() && !matches!(*name, "." | ".."))
        .unwrap_or("download");
    let encoded: String = filename
        .bytes()
        .map(|byte| format!("%{byte:02X}"))
        .collect();
    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, metadata.bytes)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"download\"; filename*=UTF-8''{encoded}"),
        )
        .header("x-content-type-options", "nosniff")
        .body(body)
        .map_err(|_| internal())
}
fn upload_timeout() -> ApiError {
    ApiError::new(
        StatusCode::REQUEST_TIMEOUT,
        "file_upload_timeout",
        "upload exceeded its 30-second input idle or 10-minute overall deadline",
    )
}

// Multer bounds file data but buffers an unfinished preamble/part header. Inspect
// the MIME framing before yielding bytes to it, retaining only matcher indexes.
// KMP keeps boundary recognition linear even for adversarial repeating payloads.
struct BoundaryMatcher {
    pattern: Vec<u8>,
    prefix: Vec<usize>,
    matched: usize,
}
impl BoundaryMatcher {
    fn new(pattern: Vec<u8>) -> Self {
        let mut prefix = vec![0; pattern.len()];
        let mut matched = 0;
        for index in 1..pattern.len() {
            while matched > 0 && pattern[index] != pattern[matched] {
                matched = prefix[matched - 1];
            }
            if pattern[index] == pattern[matched] {
                matched += 1;
            }
            prefix[index] = matched;
        }
        Self {
            pattern,
            prefix,
            matched: 0,
        }
    }
    fn push(&mut self, byte: u8) -> bool {
        while self.matched > 0 && byte != self.pattern[self.matched] {
            self.matched = self.prefix[self.matched - 1];
        }
        if byte == self.pattern[self.matched] {
            self.matched += 1;
        }
        if self.matched == self.pattern.len() {
            self.matched = self.prefix[self.matched - 1];
            true
        } else {
            false
        }
    }
}
#[derive(Clone, Copy)]
enum MultipartPhase {
    Preamble,
    Suffix(Option<u8>),
    Headers,
    Data,
    End,
}
struct MultipartGuard {
    initial: BoundaryMatcher,
    boundary: BoundaryMatcher,
    phase: MultipartPhase,
    overhead: usize,
    header_tail: u32,
    parts: usize,
}
impl MultipartGuard {
    fn new(boundary: &str) -> Result<Self, ApiError> {
        if boundary.is_empty()
            || boundary.len() > 70
            || !boundary.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
        {
            return Err(ApiError::invalid(
                "multipart boundary must contain 1..70 printable ASCII bytes",
            ));
        }
        Ok(Self {
            initial: BoundaryMatcher::new(format!("--{boundary}").into_bytes()),
            boundary: BoundaryMatcher::new(format!("\r\n--{boundary}").into_bytes()),
            phase: MultipartPhase::Preamble,
            overhead: 0,
            header_tail: 0,
            parts: 0,
        })
    }
    fn push(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let invalid = || {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "multipart preamble, header, epilogue or field count exceeds its limit, or framing is invalid",
            )
        };
        for &byte in bytes {
            match self.phase {
                MultipartPhase::Preamble => {
                    self.overhead += 1;
                    if self.overhead > MAX_PREAMBLE {
                        return Err(invalid());
                    }
                    if self.initial.push(byte) {
                        self.phase = MultipartPhase::Suffix(None);
                    }
                }
                MultipartPhase::Suffix(None) => {
                    if !matches!(byte, b'\r' | b'-') {
                        return Err(invalid());
                    }
                    self.phase = MultipartPhase::Suffix(Some(byte));
                }
                MultipartPhase::Suffix(Some(first)) => {
                    self.overhead = 0;
                    if first == b'-' && byte == b'-' {
                        self.phase = MultipartPhase::End;
                    } else if first == b'\r' && byte == b'\n' {
                        self.parts += 1;
                        if self.parts > 8 {
                            return Err(invalid());
                        }
                        self.header_tail = 0;
                        self.phase = MultipartPhase::Headers;
                    } else {
                        return Err(invalid());
                    }
                }
                MultipartPhase::Headers => {
                    self.overhead += 1;
                    if self.overhead > MAX_PART_HEADERS {
                        return Err(invalid());
                    }
                    self.header_tail = (self.header_tail << 8) | u32::from(byte);
                    if self.header_tail == u32::from_be_bytes(*b"\r\n\r\n") {
                        self.boundary.matched = 0;
                        self.phase = MultipartPhase::Data;
                    }
                }
                MultipartPhase::Data => {
                    if self.boundary.push(byte) {
                        self.phase = MultipartPhase::Suffix(None);
                    }
                }
                MultipartPhase::End => {
                    self.overhead += 1;
                    if self.overhead > MAX_PREAMBLE {
                        return Err(invalid());
                    }
                }
            }
        }
        Ok(())
    }
}
fn guard_multipart(request: Request) -> Result<Request, ApiError> {
    let boundary = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| multer::parse_boundary(value).ok())
        .ok_or_else(|| ApiError::invalid("expected multipart/form-data with a boundary"))?;
    let mut guard = MultipartGuard::new(&boundary)?;
    let (parts, body) = request.into_parts();
    let stream = body.into_data_stream().map(move |result| {
        let bytes = result.map_err(std::io::Error::other)?;
        guard.push(&bytes)?;
        Ok::<_, std::io::Error>(bytes)
    });
    Ok(Request::from_parts(parts, Body::from_stream(stream)))
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        too_large()
    } else {
        ApiError::invalid("invalid multipart form data")
    }
}
async fn small_field(field: &mut axum::extract::multipart::Field<'_>) -> Result<String, ApiError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
        if bytes.len() + chunk.len() > 1024 {
            return Err(ApiError::invalid("multipart text field is too long"));
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes).map_err(|_| ApiError::invalid("multipart text fields must be UTF-8"))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expiration {
    anchor: String,
    seconds: u64,
}
async fn upload(
    State(state): State<FileState>,
    request: Request,
) -> Result<Json<FileMetadata>, ApiError> {
    let request = guard_multipart(request)?;
    let multipart = Multipart::from_request(request, &state)
        .await
        .map_err(|_| ApiError::invalid("expected multipart/form-data with file and purpose"))?;
    tokio::time::timeout(UPLOAD_DEADLINE, upload_inner(state, multipart, UPLOAD_IDLE))
        .await
        .map_err(|_| upload_timeout())?
}

async fn upload_inner(
    state: FileState,
    mut multipart: Multipart,
    idle: Duration,
) -> Result<Json<FileMetadata>, ApiError> {
    let store = state.store()?;
    let mut pending = None;
    let mut filename = None;
    let mut purpose = None;
    let mut expiry_object = None;
    let mut expiry_anchor = None;
    let mut expiry_seconds = None;
    let mut seen = std::collections::HashSet::new();
    while let Some(mut field) = tokio::time::timeout(idle, multipart.next_field())
        .await
        .map_err(|_| upload_timeout())?
        .map_err(multipart_error)?
    {
        let name = field
            .name()
            .ok_or_else(|| ApiError::invalid("multipart fields must have names"))?
            .to_owned();
        if !matches!(
            name.as_str(),
            "file"
                | "purpose"
                | "expires_after"
                | "expires_after[anchor]"
                | "expires_after[seconds]"
        ) {
            return Err(ApiError::invalid("unknown multipart field"));
        }
        if !seen.insert(name.clone()) {
            return Err(ApiError::invalid("duplicate multipart field"));
        }
        if name == "file" {
            let name = field
                .file_name()
                .ok_or_else(|| ApiError::invalid("file field requires a filename"))?
                .to_owned();
            validate_filename(&name)?;
            let mut upload = store.begin().await?;
            let mut file = tokio::fs::File::from_std(
                upload.temp.as_file().try_clone().map_err(|_| internal())?,
            );
            while let Some(chunk) = tokio::time::timeout(idle, field.chunk())
                .await
                .map_err(|_| upload_timeout())?
                .map_err(multipart_error)?
            {
                let bytes = upload
                    .bytes
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(too_large)?;
                upload.check_size(bytes)?;
                for piece in chunk.chunks(CHUNK) {
                    file.write_all(piece).await.map_err(|_| internal())?;
                }
                upload.bytes = bytes;
            }
            file.sync_all().await.map_err(|_| internal())?;
            drop(file);
            filename = Some(name);
            pending = Some(upload);
        } else {
            if field.file_name().is_some() {
                return Err(ApiError::invalid("only file may have a filename"));
            }
            let text = tokio::time::timeout(idle, small_field(&mut field))
                .await
                .map_err(|_| upload_timeout())??;
            match name.as_str() {
                "purpose" => {
                    validate_purpose(&text, false)?;
                    purpose = Some(text);
                }
                "expires_after" => {
                    expiry_object =
                        Some(serde_json::from_str::<Expiration>(&text).map_err(|_| {
                            ApiError::invalid(
                                "expires_after must contain anchor and integer seconds",
                            )
                        })?);
                }
                "expires_after[anchor]" => expiry_anchor = Some(text),
                "expires_after[seconds]" => {
                    expiry_seconds = Some(text.parse::<u64>().map_err(|_| {
                        ApiError::invalid("expires_after.seconds must be an integer")
                    })?)
                }
                _ => unreachable!(),
            }
        }
    }
    let expiry = if let Some(object) = expiry_object {
        if expiry_anchor.is_some() || expiry_seconds.is_some() {
            return Err(ApiError::invalid("duplicate expires_after representations"));
        }
        Some(object)
    } else if expiry_anchor.is_some() || expiry_seconds.is_some() {
        Some(Expiration {
            anchor: expiry_anchor
                .ok_or_else(|| ApiError::invalid("expires_after.anchor is required"))?,
            seconds: expiry_seconds
                .ok_or_else(|| ApiError::invalid("expires_after.seconds is required"))?,
        })
    } else {
        None
    };
    if expiry
        .as_ref()
        .is_some_and(|expiry| expiry.anchor != "created_at")
    {
        return Err(ApiError::invalid("expires_after.anchor must be created_at"));
    }
    let seconds = expiry.map(|expiry| expiry.seconds);
    validate_expiry(seconds)?;
    let pending = pending.ok_or_else(|| ApiError::invalid("file field is required"))?;
    let purpose = purpose.ok_or_else(|| ApiError::invalid("purpose field is required"))?;
    Ok(Json(
        store
            .finish(pending, filename.ok_or_else(internal)?, purpose, seconds)
            .await?,
    ))
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_os = "macos")
))]
mod tests {
    use super::*;
    use axum::http::Request as HttpRequest;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn fixture() -> (tempfile::TempDir, FileStore) {
        let directory = tempfile::tempdir().unwrap();
        // macOS's system temporary directory can have a /var -> /private/var alias.
        let root = directory.path().canonicalize().unwrap().join("files");
        (directory, FileStore::new(root))
    }
    fn app(store: &FileStore, key: Option<&str>, loopback: bool) -> Router {
        file_router(FileState {
            store: Some(store.clone()),
            api_key: key.map(str::to_owned),
            loopback,
        })
    }
    fn request(method: &str, path: &str, key: Option<&str>) -> HttpRequest<Body> {
        let mut builder = HttpRequest::builder().method(method).uri(path);
        if let Some(key) = key {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        builder.body(Body::empty()).unwrap()
    }
    fn form(fields: &[(&str, Option<&str>, &[u8])], key: Option<&str>) -> HttpRequest<Body> {
        let mut bytes = Vec::new();
        for (name, filename, data) in fields {
            bytes.extend_from_slice(
                format!("--lma-boundary\r\nContent-Disposition: form-data; name=\"{name}\"")
                    .as_bytes(),
            );
            if let Some(filename) = filename {
                bytes.extend_from_slice(
                    format!("; filename=\"{filename}\"\r\nContent-Type: application/octet-stream")
                        .as_bytes(),
                );
            }
            bytes.extend_from_slice(b"\r\n\r\n");
            bytes.extend_from_slice(data);
            bytes.extend_from_slice(b"\r\n");
        }
        bytes.extend_from_slice(b"--lma-boundary--\r\n");
        let mut builder = HttpRequest::builder()
            .method("POST")
            .uri("/v1/files")
            .header(
                header::CONTENT_TYPE,
                "multipart/form-data; boundary=lma-boundary",
            );
        if let Some(key) = key {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        builder.body(Body::from(bytes)).unwrap()
    }
    async fn value(response: Response) -> Value {
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }
    fn status(error: ApiError) -> StatusCode {
        error.into_response().status()
    }

    #[test]
    fn atomic_publication_never_links_or_replaces_existing_files() {
        use std::os::unix::fs::MetadataExt;
        let (_directory, store) = fixture();
        let _lock = store.lock(true).unwrap().unwrap();
        let mut temp = tempfile::NamedTempFile::new_in(&store.root).unwrap();
        temp.write_all(b"new bytes").unwrap();
        let source = temp.path().to_owned();
        let target = store.root.join("atomic-target");
        fs::write(&target, b"existing bytes").unwrap();
        let error = publish_file(temp, &target).unwrap_err();
        assert_eq!(fs::read(&target).unwrap(), b"existing bytes");
        assert_eq!(fs::read(&source).unwrap(), b"new bytes");
        fs::remove_file(&target).unwrap();
        let file = publish_file(error.file, &target).unwrap();
        assert!(!source.exists());
        assert_eq!(file.metadata().unwrap().nlink(), 1);
        assert_eq!(fs::read(&target).unwrap(), b"new bytes");
    }

    #[tokio::test]
    async fn multipart_guard_handles_split_boundaries_and_binary_payloads() {
        let payload = b"binary\0\xff\r\n--lma-boundarX\r\n--lma-boun\0\r\nend";
        let request = form(
            &[
                ("file", Some("binary"), payload),
                ("purpose", None, b"user_data"),
            ],
            None,
        );
        let body = request.into_body().collect().await.unwrap().to_bytes();
        for chunk in [1, 2, 3, 7, 64, body.len()] {
            let mut guard = MultipartGuard::new("lma-boundary").unwrap();
            for bytes in body.chunks(chunk) {
                guard.push(bytes).unwrap();
            }
            assert!(matches!(guard.phase, MultipartPhase::End));
        }
    }

    #[tokio::test]
    async fn oversized_multipart_framing_is_rejected_before_parser_buffer_growth() {
        let (_directory, store) = fixture();
        let cases = [
            vec![b'x'; MAX_PREAMBLE + 1],
            [
                b"--lma-boundary\r\nX-Long: ".as_slice(),
                &vec![b'x'; MAX_PART_HEADERS + 1],
            ]
            .concat(),
            [
                b"--lma-boundary--\r\n".as_slice(),
                &vec![b'x'; MAX_PREAMBLE + 1],
            ]
            .concat(),
        ];
        for bytes in cases {
            let mut guard = MultipartGuard::new("lma-boundary").unwrap();
            assert!(guard.push(&bytes).is_err());
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/v1/files")
                .header(
                    header::CONTENT_TYPE,
                    "multipart/form-data; boundary=lma-boundary",
                )
                .body(Body::from(bytes))
                .unwrap();
            let response = app(&store, None, true).oneshot(req).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(value(response).await["error"].is_object());
            assert!(!store.root.exists());
        }
    }

    #[tokio::test]
    async fn stalled_upload_times_out_and_releases_private_store_lock() {
        for tail in [
            "unfinished file bytes",
            "file bytes\r\n--lma-boundary\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n",
        ] {
            let (_directory, store) = fixture();
            let (sender, receiver) =
                tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(1);
            sender.send(Ok(format!("--lma-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"safe\"\r\n\r\n{tail}").into_bytes())).await.unwrap();
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/v1/files")
                .header(
                    header::CONTENT_TYPE,
                    "multipart/form-data; boundary=lma-boundary",
                )
                .body(Body::from_stream(
                    tokio_stream::wrappers::ReceiverStream::new(receiver),
                ))
                .unwrap();
            let state = FileState {
                store: Some(store.clone()),
                api_key: None,
                loopback: true,
            };
            let multipart = Multipart::from_request(guard_multipart(req).unwrap(), &state)
                .await
                .unwrap();
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                upload_inner(state, multipart, Duration::from_millis(100)),
            )
            .await
            .unwrap();
            let response = result.unwrap_err().into_response();
            assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
            assert_eq!(
                value(response).await["error"]["code"],
                "file_upload_timeout"
            );
            drop(sender);
            let _lock = store
                .lock(false)
                .unwrap()
                .expect("upload started private staging");
            assert_eq!(store.scan_locked().unwrap().len(), 0);
            assert!(fs::read_dir(&store.root).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".upload-")
            }));
        }
    }

    #[tokio::test]
    async fn construction_and_missing_list_do_not_create_directories() {
        let (_directory, store) = fixture();
        let list = store.list(ListOptions::default()).await.unwrap();
        assert_eq!(list["data"], json!([]));
        assert_eq!(list["first_id"], Value::Null);
        assert_eq!(list["last_id"], Value::Null);
        assert!(!store.root.exists());
        assert_eq!(
            status(store.metadata("file-abcdef").await.unwrap_err()),
            StatusCode::NOT_FOUND
        );
        assert!(!store.root.exists());
    }

    #[tokio::test]
    async fn multipart_binary_order_metadata_content_head_delete() {
        let (_directory, store) = fixture();
        let app = app(&store, None, true);
        let binary = b"\0\xff\xfe\r\nnot utf8\0";
        for file_first in [true, false] {
            let mut fields: Vec<(&str, Option<&str>, &[u8])> = vec![
                ("file", Some("../../binary.bin"), binary),
                ("purpose", None, b"user_data"),
            ];
            if !file_first {
                fields.reverse();
            }
            let response = app.clone().oneshot(form(&fields, None)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let uploaded = value(response).await;
            assert_eq!(uploaded["object"], "file");
            assert_eq!(uploaded["status"], "uploaded");
            assert_eq!(uploaded["purpose"], "user_data");
            assert_eq!(uploaded["bytes"], binary.len());
            assert_eq!(uploaded["filename"], "../../binary.bin");
            assert!(uploaded["created_at"].as_u64().unwrap() <= now());
            let id = uploaded["id"].as_str().unwrap();
            validate_id(id).unwrap();
            let path = format!("/v1/files/{id}");
            let response = app
                .clone()
                .oneshot(request("GET", &path, None))
                .await
                .unwrap();
            assert_eq!(value(response).await, uploaded);
            let content_path = format!("{path}/content");
            let response = app
                .clone()
                .oneshot(request("GET", &content_path, None))
                .await
                .unwrap();
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                "application/octet-stream"
            );
            assert_eq!(response.headers()["x-content-type-options"], "nosniff");
            assert!(
                response.headers()[header::CONTENT_DISPOSITION]
                    .to_str()
                    .unwrap()
                    .starts_with("attachment;")
            );
            assert_eq!(
                &response.into_body().collect().await.unwrap().to_bytes()[..],
                binary
            );
            for path in ["/v1/files".to_owned(), path.clone(), content_path] {
                let response = app
                    .clone()
                    .oneshot(request("HEAD", &path, None))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert!(
                    response
                        .into_body()
                        .collect()
                        .await
                        .unwrap()
                        .to_bytes()
                        .is_empty()
                );
            }
            let response = app
                .clone()
                .oneshot(request("DELETE", &path, None))
                .await
                .unwrap();
            assert_eq!(
                value(response).await,
                json!({"id":id,"object":"file","deleted":true})
            );
            assert_eq!(
                app.clone()
                    .oneshot(request("GET", &path, None))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::NOT_FOUND
            );
            assert!(!store.root.join(format!("{id}.data")).exists());
            assert!(!store.root.join(format!("{id}.json")).exists());
        }
    }

    #[tokio::test]
    async fn rejects_missing_duplicate_unknown_and_invalid_multipart_fields() {
        let (_directory, store) = fixture();
        let app = app(&store, None, true);
        let cases: Vec<Vec<(&str, Option<&str>, &[u8])>> = vec![
            vec![],
            vec![("purpose", None, b"batch")],
            vec![("file", Some("a"), b"data")],
            vec![("file", None, b"data"), ("purpose", None, b"batch")],
            vec![
                ("file", Some("a"), b"data"),
                ("purpose", None, b"batch"),
                ("file", Some("b"), b"data"),
            ],
            vec![
                ("file", Some("a"), b"data"),
                ("purpose", None, b"batch"),
                ("purpose", None, b"batch"),
            ],
            vec![
                ("file", Some("a"), b"data"),
                ("purpose", None, b"batch_output"),
            ],
            vec![
                ("file", Some("a"), b"data"),
                ("purpose", None, b"not-a-purpose"),
            ],
            vec![
                ("file", Some("a"), b"data"),
                ("purpose", None, b"batch"),
                ("unknown", None, b"secret"),
            ],
            vec![
                ("file", Some("a"), b"data"),
                ("purpose", Some("x"), b"batch"),
            ],
            vec![("file", Some(""), b"data"), ("purpose", None, b"batch")],
            vec![("file", Some("a"), b"data"), ("purpose", None, b"\xff")],
        ];
        for fields in cases {
            let response = app.clone().oneshot(form(&fields, None)).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = value(response).await;
            assert!(body["error"]["message"].is_string());
            assert!(!body.to_string().contains("secret"));
            assert_eq!(
                store.list(ListOptions::default()).await.unwrap()["data"],
                json!([])
            );
        }
        let response = app
            .oneshot(request("POST", "/v1/files", None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(value(response).await["error"].is_object());
        let long_name = "x".repeat(MAX_FILENAME + 1);
        assert!(
            store
                .store_bytes(&long_name, "batch", b"a", None)
                .await
                .is_err()
        );
        let long_purpose = vec![b'x'; 1025];
        let response = self::app(&store, None, true)
            .oneshot(form(&[("purpose", None, &long_purpose)], None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upload_purposes_and_expiration_match_contract() {
        let (_directory, store) = fixture();
        let app = app(&store, None, true);
        for purpose in UPLOAD_PURPOSES {
            let response = app
                .clone()
                .oneshot(form(
                    &[
                        ("purpose", None, purpose.as_bytes()),
                        ("file", Some("a.bin"), b"bytes"),
                    ],
                    None,
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let file = value(response).await;
            assert_eq!(file["purpose"], *purpose);
            if *purpose == "batch" {
                assert_eq!(
                    file["expires_at"].as_u64().unwrap() - file["created_at"].as_u64().unwrap(),
                    2_592_000
                );
            } else {
                assert!(file.get("expires_at").is_none());
            }
        }
        for fields in [
            vec![
                ("expires_after[seconds]", None, b"3600".as_slice()),
                ("expires_after[anchor]", None, b"created_at".as_slice()),
            ],
            vec![(
                "expires_after",
                None,
                br#"{"anchor":"created_at","seconds":3600}"#.as_slice(),
            )],
        ] {
            let mut fields = fields;
            fields.extend([
                ("file", Some("a"), b"a".as_slice()),
                ("purpose", None, b"user_data".as_slice()),
            ]);
            let response = app.clone().oneshot(form(&fields, None)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let file = value(response).await;
            assert_eq!(
                file["expires_at"].as_u64().unwrap() - file["created_at"].as_u64().unwrap(),
                3600
            );
        }
        for expiry in [
            br#"{"anchor":"created_at","seconds":3599}"#.as_slice(),
            br#"{"anchor":"created_at","seconds":2592001}"#.as_slice(),
            br#"{"anchor":"other","seconds":3600}"#.as_slice(),
            br#"{"anchor":"created_at","seconds":3600,"unknown":1}"#.as_slice(),
        ] {
            let response = app
                .clone()
                .oneshot(form(&[("expires_after", None, expiry)], None))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        for fields in [
            vec![("expires_after[anchor]", None, b"created_at".as_slice())],
            vec![("expires_after[seconds]", None, b"3600".as_slice())],
            vec![("expires_after[seconds]", None, b"1.5".as_slice())],
            vec![
                (
                    "expires_after",
                    None,
                    br#"{"anchor":"created_at","seconds":3600}"#.as_slice(),
                ),
                ("expires_after[seconds]", None, b"3600".as_slice()),
            ],
            vec![
                ("expires_after[anchor]", None, b"created_at".as_slice()),
                ("expires_after[anchor]", None, b"created_at".as_slice()),
            ],
        ] {
            let response = app.clone().oneshot(form(&fields, None)).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn pagination_filters_and_cursor_are_stable_and_validated() {
        let (_directory, store) = fixture();
        let mut files = Vec::new();
        for purpose in ["batch", "user_data", "batch", "batch"] {
            files.push(store.store_bytes("a", purpose, b"abc", None).await.unwrap());
        }
        files.sort_by(|a, b| (a.created_at, &a.id).cmp(&(b.created_at, &b.id)));
        for order in ["asc", "desc"] {
            let mut expected = files.clone();
            if order == "desc" {
                expected.reverse();
            }
            let mut after = None;
            for (index, file) in expected.iter().enumerate() {
                let page = store
                    .list(ListOptions {
                        limit: 1,
                        order: order.into(),
                        after,
                        ..Default::default()
                    })
                    .await
                    .unwrap();
                assert_eq!(page["data"][0]["id"], file.id);
                assert_eq!(page["first_id"], file.id);
                assert_eq!(page["last_id"], file.id);
                assert_eq!(page["has_more"], index + 1 < expected.len());
                after = Some(file.id.clone());
            }
        }
        let list = store
            .list(ListOptions {
                purpose: Some("batch".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(list["data"].as_array().unwrap().len(), 3);
        let unfiltered_cursor = files.iter().find(|f| f.purpose == "user_data").unwrap();
        assert!(
            store
                .list(ListOptions {
                    purpose: Some("batch".into()),
                    after: Some(unfiltered_cursor.id.clone()),
                    ..Default::default()
                })
                .await
                .is_ok()
        );
        let app = app(&store, None, true);
        for query in [
            "limit=0",
            "limit=10001",
            "order=sideways",
            "limit=1&limit=2",
            "unknown=1",
            "limit=NaN",
        ] {
            let response = app
                .clone()
                .oneshot(request("GET", &format!("/v1/files?{query}"), None))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
            assert!(value(response).await["error"].is_object());
        }
        let response = app
            .oneshot(request("GET", "/v1/files?after=file-missing", None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn authentication_applies_to_every_file_route_and_head() {
        let (_directory, store) = fixture();
        let app = app(&store, Some("private-key"), false);
        for (method, path) in [
            ("GET", "/v1/files"),
            ("HEAD", "/v1/files"),
            ("POST", "/v1/files"),
            ("GET", "/v1/files/file-abcdef"),
            ("DELETE", "/v1/files/file-abcdef"),
            ("GET", "/v1/files/file-abcdef/content"),
            ("HEAD", "/v1/files/file-abcdef/content"),
        ] {
            for key in [None, Some("wrong"), Some("private-key-longer")] {
                let response = app
                    .clone()
                    .oneshot(request(method, path, key))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
                assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
                assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
                if method == "HEAD" {
                    assert!(
                        response
                            .into_body()
                            .collect()
                            .await
                            .unwrap()
                            .to_bytes()
                            .is_empty()
                    );
                } else {
                    assert_eq!(value(response).await["error"]["code"], "invalid_api_key");
                }
            }
        }
        assert!(!store.root.exists());
        let response = app
            .clone()
            .oneshot(request("GET", "/v1/files", Some("private-key")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut request = request("GET", "/v1/files", None);
        request
            .headers_mut()
            .insert(header::AUTHORIZATION, "bEaReR private-key".parse().unwrap());
        assert_eq!(app.oneshot(request).await.unwrap().status(), StatusCode::OK);
        assert!(!store.root.exists());
        assert!(constant_time_equal(b"x", b"x"));
        assert!(!constant_time_equal(b"x", b"xy"));
    }

    #[tokio::test]
    async fn remote_without_key_denies_all_file_access_even_with_supplied_bearer() {
        let (_directory, store) = fixture();
        let app = app(&store, None, false);
        for (method, path) in [
            ("GET", "/v1/files"),
            ("HEAD", "/v1/files"),
            ("POST", "/v1/files"),
            ("GET", "/v1/files/file-abcdef"),
            ("DELETE", "/v1/files/file-abcdef"),
            ("GET", "/v1/files/file-abcdef/content"),
            ("HEAD", "/v1/files/file-abcdef/content"),
        ] {
            assert_eq!(
                app.clone()
                    .oneshot(request(method, path, Some("anything")))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::FORBIDDEN
            );
        }
        assert!(!store.root.exists());
        let invalid_key_app = self::app(&store, Some(""), true);
        assert_eq!(
            invalid_key_app
                .oneshot(request("GET", "/v1/files", Some("")))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn bounded_reads_generated_outputs_and_filename_escaping() {
        let (_directory, store) = fixture();
        let metadata = store
            .store_bytes(
                "../quoted\"\\file.jsonl",
                "batch_output",
                b"output\0\xff",
                Some(3600),
            )
            .await
            .unwrap();
        assert_eq!(store.read(&metadata.id, 8).await.unwrap(), b"output\0\xff");
        assert_eq!(
            status(store.read(&metadata.id, 7).await.unwrap_err()),
            StatusCode::BAD_REQUEST
        );
        let encoded = serde_json::to_string(&metadata).unwrap();
        assert_eq!(
            serde_json::from_str::<FileMetadata>(&encoded)
                .unwrap()
                .filename,
            metadata.filename
        );
        let response = app(&store, None, true)
            .oneshot(request(
                "GET",
                &format!("/v1/files/{}/content", metadata.id),
                None,
            ))
            .await
            .unwrap();
        let disposition = response.headers()[header::CONTENT_DISPOSITION]
            .to_str()
            .unwrap();
        assert!(disposition.contains("filename*=UTF-8''"));
        assert!(!disposition.contains("%2F"));
        assert!(!disposition.contains("%5C"));
        assert!(!disposition.contains("../"));
        assert!(
            store
                .store_bytes("control\nname", "user_data", b"a", None)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn small_fixture_limits_test_file_quota_and_failure_preservation() {
        let (_directory, mut store) = fixture();
        store.max_file = 8;
        store.max_store = 10;
        let first = store
            .store_bytes("first", "user_data", b"123456", None)
            .await
            .unwrap();
        assert_eq!(
            status(
                store
                    .store_bytes("large", "user_data", b"123456789", None)
                    .await
                    .unwrap_err()
            ),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            status(
                store
                    .store_bytes("quota", "user_data", b"12345", None)
                    .await
                    .unwrap_err()
            ),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(store.read(&first.id, 8).await.unwrap(), b"123456");
        let response = app(&store, None, true)
            .oneshot(form(
                &[
                    ("file", Some("a"), b"123456789"),
                    ("purpose", None, b"user_data"),
                ],
                None,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            store.list(ListOptions::default()).await.unwrap()["data"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        store.max_files = 1;
        assert_eq!(
            status(
                store
                    .store_bytes("count", "user_data", b"", None)
                    .await
                    .unwrap_err()
            ),
            StatusCode::BAD_REQUEST
        );
        store.delete(&first.id).await.unwrap();
        assert!(
            store
                .store_bytes("after-delete", "user_data", b"12345678", None)
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn expired_files_are_hidden_and_reclaimed_on_next_mutation() {
        let (_directory, mut store) = fixture();
        store.max_store = 8;
        let mut file = store
            .store_bytes("old", "batch", b"12345678", None)
            .await
            .unwrap();
        file.created_at = now() - 7200;
        file.expires_at = Some(now() - 1);
        fs::write(
            store.root.join(format!("{}.json", file.id)),
            serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
        assert_eq!(
            status(store.metadata(&file.id).await.unwrap_err()),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            store.list(ListOptions::default()).await.unwrap()["data"],
            json!([])
        );
        assert!(store.open_content(&file.id).await.is_err());
        assert!(
            store
                .store_bytes("new", "batch", b"12345678", None)
                .await
                .is_ok()
        );
        assert!(!store.root.join(format!("{}.data", file.id)).exists());
    }

    #[tokio::test]
    async fn traversal_and_malformed_ids_never_access_other_files() {
        let (_directory, store) = fixture();
        let file = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        for id in [
            "../escape",
            "/tmp/file",
            "file-../../escape",
            "file-x/y",
            "file-abc_def",
            "file-abc%2fdef",
            "file-",
            "",
        ] {
            assert!(store.metadata(id).await.is_err());
            assert!(store.delete(id).await.is_err());
        }
        let app = app(&store, None, true);
        let response = app
            .oneshot(request("GET", "/v1/files/file-abc%2Fdef", None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(store.read(&file.id, 4).await.unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_hardlink_and_unsafe_permissions_are_rejected_without_damage() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let (directory, store) = fixture();
        let outside = directory.path().join("outside");
        fs::write(&outside, b"do not change").unwrap();
        let file = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let data = store.root.join(format!("{}.data", file.id));
        fs::remove_file(&data).unwrap();
        symlink(&outside, &data).unwrap();
        assert!(store.metadata(&file.id).await.is_err());
        assert!(store.delete(&file.id).await.is_err());
        assert!(
            store
                .store_bytes("other", "batch", b"x", None)
                .await
                .is_err()
        );
        assert_eq!(fs::read(&outside).unwrap(), b"do not change");
        fs::remove_file(&data).unwrap();
        fs::hard_link(&outside, &data).unwrap();
        assert!(store.open_content(&file.id).await.is_err());
        fs::remove_file(&data).unwrap();
        fs::write(&data, b"keep").unwrap();
        fs::set_permissions(&data, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(store.read(&file.id, 4).await.unwrap(), b"keep");
        let lock = store.root.join(".files.lock");
        fs::remove_file(&lock).unwrap();
        symlink(&outside, &lock).unwrap();
        assert!(store.list(ListOptions::default()).await.is_err());
        fs::remove_file(lock).unwrap();
        fs::set_permissions(&store.root, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(store.list(ListOptions::default()).await.is_err());
        fs::set_permissions(&store.root, fs::Permissions::from_mode(0o700)).unwrap();
        let alias = directory.path().canonicalize().unwrap().join("alias");
        symlink(&store.root, &alias).unwrap();
        assert!(
            FileStore::new(alias)
                .store_bytes("bad", "batch", b"x", None)
                .await
                .is_err()
        );
        let metadata = fs::metadata(&data).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_metadata_preserves_owned_bytes_on_delete_failure() {
        use std::os::unix::fs::symlink;
        let (directory, store) = fixture();
        let metadata = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let path = store.root.join(format!("{}.json", metadata.id));
        let outside = directory.path().join("outside.json");
        fs::rename(&path, &outside).unwrap();
        symlink(&outside, &path).unwrap();
        assert!(store.delete(&metadata.id).await.is_err());
        assert_eq!(
            fs::read(store.root.join(format!("{}.data", metadata.id))).unwrap(),
            b"keep"
        );
        assert_eq!(
            serde_json::from_slice::<FileMetadata>(&fs::read(&outside).unwrap())
                .unwrap()
                .id,
            metadata.id
        );
    }

    #[tokio::test]
    async fn publication_collision_rolls_back_data_without_replacing_metadata() {
        let (_directory, store) = fixture();
        let pending = store.begin().await.unwrap();
        let suffix = pending
            .temp
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix(".upload-")
            .unwrap();
        let id = format!("file-{suffix}");
        let path = store.root.join(format!("{id}.json"));
        fs::write(&path, b"preserve existing metadata").unwrap();
        assert!(
            store
                .finish(pending, "a".into(), "user_data".into(), None)
                .await
                .is_err()
        );
        assert_eq!(fs::read(path).unwrap(), b"preserve existing metadata");
        assert!(!store.root.join(format!("{id}.data")).exists());
    }

    #[tokio::test]
    async fn failed_publication_rollback_preserves_metadata_and_data_for_recovery() {
        let (_directory, store) = fixture();
        let metadata = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let _lock = store.lock(false).unwrap().unwrap();
        let path = store.root.join(format!("{}.json", metadata.id));
        let mut marker = tempfile::Builder::new()
            .prefix(".metadata-")
            .tempfile_in(&store.root)
            .unwrap();
        marker.write_all(&fs::read(&path).unwrap()).unwrap();
        fs::remove_file(&path).unwrap();
        let marker_path = marker.path().to_owned();
        let data = store.root.join(format!("{}.data", metadata.id));
        let error = FileStore::rollback_publication(marker, &data, |_| {
            Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
        });
        assert_eq!(status(error), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(marker_path.exists());
        assert_eq!(fs::read(&data).unwrap(), b"keep");
        store.clean_staging_locked().unwrap();
        assert!(!marker_path.exists());
        assert_eq!(
            store.metadata_locked(&metadata.id, false).unwrap().filename,
            "safe"
        );
    }

    #[tokio::test]
    async fn interrupted_delete_restores_metadata_without_deleting_uploaded_bytes() {
        let (_directory, store) = fixture();
        let metadata = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let _lock = store.lock(false).unwrap().unwrap();
        let path = store.root.join(format!("{}.json", metadata.id));
        let marker = store.root.join(".deleted-interrupted");
        fs::rename(&path, &marker).unwrap();
        store.clean_staging_locked().unwrap();
        assert!(!marker.exists());
        assert_eq!(
            store.metadata_locked(&metadata.id, false).unwrap().id,
            metadata.id
        );
        assert_eq!(
            fs::read(store.root.join(format!("{}.data", metadata.id))).unwrap(),
            b"keep"
        );
    }

    #[tokio::test]
    async fn recovery_conflicts_preserve_both_versions_and_block_mutation() {
        let (_directory, store) = fixture();
        let metadata = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let _lock = store.lock(false).unwrap().unwrap();
        let path = store.root.join(format!("{}.json", metadata.id));
        let marker = store.root.join(".deleted-conflict");
        let original = fs::read(&path).unwrap();
        fs::rename(&path, &marker).unwrap();
        fs::write(&path, b"different metadata: preserve me").unwrap();
        assert!(store.restore_marker_locked(&marker, &metadata.id).is_err());
        assert!(store.clean_staging_locked().is_err());
        assert_eq!(fs::read(&marker).unwrap(), original);
        assert_eq!(fs::read(&path).unwrap(), b"different metadata: preserve me");
        assert_eq!(
            fs::read(store.root.join(format!("{}.data", metadata.id))).unwrap(),
            b"keep"
        );
    }

    #[tokio::test]
    async fn completed_delete_journal_is_reclaimed_but_invalid_journal_is_preserved() {
        let (_directory, store) = fixture();
        let metadata = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let _lock = store.lock(false).unwrap().unwrap();
        let path = store.root.join(format!("{}.json", metadata.id));
        let marker = store.root.join(".deleted-completed");
        fs::rename(&path, &marker).unwrap();
        fs::remove_file(store.root.join(format!("{}.data", metadata.id))).unwrap();
        store.clean_staging_locked().unwrap();
        assert!(!marker.exists());
        assert_eq!(store.scan_locked().unwrap().len(), 0);
        fs::write(&marker, b"damaged recovery metadata").unwrap();
        assert!(store.clean_staging_locked().is_err());
        assert_eq!(fs::read(marker).unwrap(), b"damaged recovery metadata");
    }

    #[tokio::test]
    async fn concurrent_publish_uses_independent_handle_locks_and_unique_ids() {
        let (_directory, store) = fixture();
        let pending = store.begin().await.unwrap();
        assert_eq!(
            status(
                store
                    .store_bytes("busy", "user_data", b"a", None)
                    .await
                    .unwrap_err()
            ),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(pending);
        let mut tasks = Vec::new();
        for index in 0..8u8 {
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                for _ in 0..100 {
                    match store
                        .store_bytes("concurrent", "user_data", &[index], None)
                        .await
                    {
                        Ok(file) => return file,
                        Err(error) => {
                            assert_eq!(status(error), StatusCode::TOO_MANY_REQUESTS);
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        }
                    }
                }
                panic!("lock never became available");
            }));
        }
        let mut ids = std::collections::HashSet::new();
        let mut published = Vec::new();
        for (index, task) in tasks.into_iter().enumerate() {
            let file = task.await.unwrap();
            assert!(ids.insert(file.id.clone()));
            assert_eq!(file.bytes, 1);
            published.push((index as u8, file));
        }
        // Other publishers may own the lock until all tasks have been joined.
        for (index, file) in published {
            assert_eq!(store.read(&file.id, 1).await.unwrap(), vec![index]);
        }
        assert_eq!(
            store.list(ListOptions::default()).await.unwrap()["data"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
    }

    #[tokio::test]
    async fn publication_never_overwrites_existing_data() {
        let (_directory, store) = fixture();
        let pending = store.begin().await.unwrap();
        let suffix = pending
            .temp
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix(".upload-")
            .unwrap();
        let id = format!("file-{suffix}");
        let path = store.root.join(format!("{id}.data"));
        fs::write(&path, b"preserve existing data").unwrap();
        assert!(
            store
                .finish(pending, "a".into(), "user_data".into(), None)
                .await
                .is_err()
        );
        assert_eq!(fs::read(path).unwrap(), b"preserve existing data");
        assert!(!store.root.join(format!("{id}.json")).exists());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[tokio::test]
    async fn fifo_data_is_rejected_without_opening_or_blocking() {
        use std::{
            ffi::CString,
            os::unix::{ffi::OsStrExt, fs::FileTypeExt},
        };
        #[cfg(target_os = "macos")]
        type Mode = u16;
        #[cfg(target_os = "linux")]
        type Mode = u32;
        unsafe extern "C" {
            fn mkfifo(path: *const std::ffi::c_char, mode: Mode) -> i32;
        }
        let (_directory, store) = fixture();
        let file = store
            .store_bytes("safe", "user_data", b"keep", None)
            .await
            .unwrap();
        let path = store.root.join(format!("{}.data", file.id));
        fs::remove_file(&path).unwrap();
        let encoded = CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: encoded is a live, NUL-terminated path; mode has the OS ABI type.
        assert_eq!(unsafe { mkfifo(encoded.as_ptr(), 0o600) }, 0);
        assert!(store.open_content(&file.id).await.is_err());
        assert!(store.delete(&file.id).await.is_err());
        assert!(fs::symlink_metadata(&path).unwrap().file_type().is_fifo());
    }

    #[tokio::test]
    async fn abandoned_staging_is_reclaimed_and_corrupt_metadata_fails_closed() {
        let (_directory, store) = fixture();
        let pending = store.begin().await.unwrap();
        let temp = tempfile::Builder::new()
            .prefix(".upload-")
            .tempfile_in(&store.root)
            .unwrap();
        let (_, path) = temp.keep().unwrap();
        drop(pending);
        assert!(path.exists());
        let metadata = store
            .store_bytes("next", "user_data", b"next", None)
            .await
            .unwrap();
        assert!(!path.exists());
        fs::write(
            store.root.join(format!("{}.json", metadata.id)),
            b"corrupt secret data",
        )
        .unwrap();
        let response = app(&store, None, true)
            .oneshot(request("GET", "/v1/files", None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!value(response).await.to_string().contains("secret"));
    }
}
