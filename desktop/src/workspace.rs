//! Desktop-local chat storage. Call these blocking APIs off the async runtime.
//! The app-config directory is supplied by the desktop host, never by the webview.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const FILE_NAME: &str = "desktop-workspace.json";
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_TEXT: usize = 1024 * 1024;
const MAX_IMAGE: usize = 8 * 1024 * 1024;
const MAX_ENCODED_IMAGE: usize = MAX_IMAGE.div_ceil(3) * 4;
static STORAGE: Mutex<()> = Mutex::new(());
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

type Result<T> = std::result::Result<T, String>;

fn empty_workspace() -> Value {
    serde_json::json!({"version": 1, "currentId": null, "conversations": []})
}

/// Read without creating directories or files, including when nothing is saved.
pub fn load(config_dir: &Path) -> Result<Value> {
    let _guard = STORAGE
        .lock()
        .map_err(|_| "Workspace storage lock failed.")?;
    load_inner(config_dir)
}

/// Validate first, then atomically replace only an absent or valid workspace.
/// Errors never contain conversation text, image data, or unrecognized keys.
pub fn save(config_dir: &Path, workspace: Value) -> Result<()> {
    let _guard = STORAGE
        .lock()
        .map_err(|_| "Workspace storage lock failed.")?;
    validate(&workspace)?;
    let bytes = serialize(&workspace)?;
    load_inner(config_dir)?;
    fs::create_dir_all(config_dir)
        .map_err(|e| format!("Cannot create workspace directory: {e}"))?;
    let (mut temporary, mut file) = create_temporary(config_dir)?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| format!("Cannot write workspace: {e}"))?;
    drop(file);
    // Also reject an unsafe destination installed while the temporary was written.
    load_inner(config_dir)?;
    fs::rename(&temporary.path, config_dir.join(FILE_NAME))
        .map_err(|e| format!("Cannot replace workspace: {e}"))?;
    temporary.committed = true;
    Ok(())
}

fn load_inner(config_dir: &Path) -> Result<Value> {
    let path = lma::models::storage::read_path(config_dir, FILE_NAME)?;
    let before = match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(empty_workspace()),
        Err(e) => return Err(format!("Cannot inspect saved workspace: {e}")),
        Ok(metadata) => metadata,
    };
    if !before.file_type().is_file() {
        return Err("Saved workspace must be a regular, non-symlink file.".into());
    }
    if before.len() > MAX_BYTES as u64 {
        return Err("Saved workspace exceeds 64 MiB.".into());
    }
    let file = open_readonly(&path)?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("Cannot inspect workspace: {e}"))?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_BYTES as u64 {
        return Err("Saved workspace must be a regular file of at most 64 MiB.".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != metadata.dev() || before.ino() != metadata.ino() {
            return Err("Saved workspace changed while opening it.".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Cannot read saved workspace: {e}"))?;
    if bytes.len() > MAX_BYTES {
        return Err("Saved workspace exceeds 64 MiB.".into());
    }
    let workspace =
        serde_json::from_slice(&bytes).map_err(|_| "Saved workspace contains invalid JSON.")?;
    validate(&workspace)?;
    Ok(workspace)
}

fn open_readonly(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    // The metadata check alone cannot prevent a symlink/FIFO replacement race.
    // These are the OS O_NOFOLLOW | O_NONBLOCK flags; no additional libc dependency.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
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
        // FILE_FLAG_OPEN_REPARSE_POINT: inspect the link itself, not its target.
        options.custom_flags(0x00200000);
    }
    options
        .open(path)
        .map_err(|e| format!("Cannot open saved workspace: {e}"))
}

struct Temporary {
    path: PathBuf,
    committed: bool,
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn create_temporary(config_dir: &Path) -> Result<(Temporary, File)> {
    for _ in 0..64 {
        let path = config_dir.join(format!(
            ".desktop-workspace-{}-{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        match create_temporary_at(path) {
            Ok(temporary) => return Ok(temporary),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!(
                    "Cannot create private workspace temporary file: {e}"
                ));
            }
        }
    }
    Err("Cannot allocate a workspace temporary file.".into())
}

fn create_temporary_at(path: PathBuf) -> io::Result<(Temporary, File)> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    // Never construct the cleanup guard until this invocation owns the file.
    let file = options.open(&path)?;
    let temporary = Temporary {
        path,
        committed: false,
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok((temporary, file))
}

struct LimitedWriter {
    bytes: Vec<u8>,
}

impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_BYTES - self.bytes.len() {
            return Err(io::Error::other("Workspace exceeds 64 MiB."));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialize(workspace: &Value) -> Result<Vec<u8>> {
    let mut writer = LimitedWriter { bytes: Vec::new() };
    serde_json::to_writer(&mut writer, workspace)
        .map_err(|_| "Cannot serialize workspace within the 64 MiB limit.")?;
    Ok(writer.bytes)
}

fn object<'a>(
    value: &'a Value,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or("Workspace field must be an object.")?;
    if required.iter().any(|key| !object.contains_key(*key))
        || object
            .keys()
            .any(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err("Workspace object has missing or unrecognized fields.".into());
    }
    Ok(object)
}

fn text(value: &Value, maximum: usize) -> Result<&str> {
    let text = value
        .as_str()
        .ok_or("Workspace text field must be a string.")?;
    if text.len() > maximum {
        return Err("Workspace text field exceeds its byte limit.".into());
    }
    Ok(text)
}

fn array(value: &Value, maximum: usize) -> Result<&[Value]> {
    let array = value
        .as_array()
        .ok_or("Workspace field must be an array.")?;
    if array.len() > maximum {
        return Err("Workspace array exceeds its item limit.".into());
    }
    Ok(array)
}

fn unsigned(value: &Value, maximum: u64) -> Result<u64> {
    value
        .as_u64()
        .filter(|n| *n <= maximum)
        .ok_or_else(|| "Workspace integer is outside its permitted range.".into())
}

fn number(value: &Value, min: f64, max: f64, exclusive_min: bool) -> Result<()> {
    let n = value
        .as_f64()
        .ok_or("Workspace field must be a finite number.")?;
    if !n.is_finite() || n < min || n > max || (exclusive_min && n == min) {
        return Err("Workspace number is outside its permitted range.".into());
    }
    Ok(())
}

fn mode(value: &Value) -> Result<()> {
    match value.as_str() {
        Some("chat" | "raw") => Ok(()),
        _ => Err("Workspace mode must be chat or raw.".into()),
    }
}

fn settings(value: &Value) -> Result<()> {
    let s = object(
        value,
        &[
            "mode",
            "system",
            "coding",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        ],
        &[],
    )?;
    mode(&s["mode"])?;
    text(&s["system"], MAX_TEXT)?;
    if !s["coding"].is_boolean() {
        return Err("Workspace coding setting must be boolean.".into());
    }
    if unsigned(&s["max_tokens"], u32::MAX as u64)? == 0 {
        return Err("Workspace token limit must be positive.".into());
    }
    number(&s["temperature"], 0.0, 2.0, false)?;
    number(&s["top_p"], 0.0, 1.0, true)?;
    if s["seed"].as_str() != Some("") {
        unsigned(&s["seed"], u32::MAX as u64)?;
    }
    Ok(())
}

fn images(value: &Value) -> Result<()> {
    for image in array(value, 8)? {
        let image = object(image, &["base64", "mime", "name", "size"], &[])?;
        text(&image["name"], MAX_TEXT)?;
        let size = unsigned(&image["size"], MAX_IMAGE as u64)? as usize;
        let encoded = text(&image["base64"], MAX_ENCODED_IMAGE)?;
        // STANDARD rejects noncanonical padding, trailing bits, whitespace and data URLs.
        let decoded = STANDARD
            .decode(encoded)
            .map_err(|_| "Workspace image has invalid base64.")?;
        if decoded.len() > MAX_IMAGE || decoded.len() != size {
            return Err("Workspace image size does not match its data.".into());
        }
        let matches_mime = match image["mime"].as_str() {
            Some("image/png") => decoded.starts_with(b"\x89PNG\r\n\x1a\n"),
            Some("image/jpeg") => decoded.starts_with(b"\xff\xd8\xff"),
            _ => false,
        };
        if !matches_mime {
            return Err("Workspace image MIME type does not match a PNG/JPEG signature.".into());
        }
    }
    Ok(())
}

fn message(value: &Value) -> Result<()> {
    let m = object(
        value,
        &["role", "content"],
        &[
            "images", "mode", "settings", "elapsed", "usage", "finish", "model",
        ],
    )?;
    if !matches!(m["role"].as_str(), Some("user" | "assistant")) {
        return Err("Workspace message role must be user or assistant.".into());
    }
    text(&m["content"], MAX_TEXT)?;
    if let Some(value) = m.get("images") {
        images(value)?;
    }
    if let Some(value) = m.get("mode") {
        mode(value)?;
    }
    if let Some(value) = m.get("settings") {
        settings(value)?;
    }
    if let Some(value) = m.get("elapsed") {
        number(value, 0.0, f64::MAX, false)?;
    }
    if let Some(value) = m.get("usage") {
        let usage = object(
            value,
            &[],
            &["prompt_tokens", "completion_tokens", "total_tokens"],
        )?;
        for count in usage.values() {
            unsigned(count, usize::MAX as u64)?;
        }
    }
    for key in ["finish", "model"] {
        if let Some(value) = m.get(key) {
            text(value, MAX_TEXT)?;
        }
    }
    Ok(())
}

fn validate(workspace: &Value) -> Result<()> {
    let root = object(workspace, &["version", "currentId", "conversations"], &[])?;
    if root["version"].as_u64() != Some(1) {
        return Err("Unsupported workspace version.".into());
    }
    let current = if root["currentId"].is_null() {
        None
    } else {
        Some(text(&root["currentId"], 128)?)
    };
    let mut ids = HashSet::new();
    let mut total_messages = 0;
    for conversation in array(&root["conversations"], 100)? {
        let c = object(
            conversation,
            &[
                "id",
                "title",
                "createdAt",
                "updatedAt",
                "settings",
                "draft",
                "attachments",
                "messages",
            ],
            &[],
        )?;
        let id = text(&c["id"], 128)?;
        if id.is_empty() || !ids.insert(id) {
            return Err("Workspace conversation IDs must be nonempty and unique.".into());
        }
        text(&c["title"], 256)?;
        text(&c["createdAt"], 64)?;
        text(&c["updatedAt"], 64)?;
        settings(&c["settings"])?;
        text(&c["draft"], MAX_TEXT)?;
        images(&c["attachments"])?;
        let messages = array(&c["messages"], 1000)?;
        total_messages += messages.len();
        if total_messages > 10_000 {
            return Err("Workspace exceeds 10000 total messages.".into());
        }
        for value in messages {
            message(value)?;
        }
    }
    if current.is_some_and(|id| !ids.contains(id)) {
        return Err("Workspace currentId does not reference a conversation.".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let root = std::env::temp_dir().join(format!(
                    "lma-workspace-test-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&root) {
                    Ok(()) => return Self(root),
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("Cannot create test directory: {e}"),
                }
            }
        }
        fn path(&self) -> PathBuf {
            self.0.join(FILE_NAME)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn example_settings() -> Value {
        json!({"mode":"raw", "system":"Be precise.\n```rust\nfn main() {}\n```", "coding":true,
            "max_tokens":2048, "temperature":0.7, "top_p":0.95, "seed":42})
    }
    fn png() -> Value {
        let bytes = b"\x89PNG\r\n\x1a\nfixture";
        json!({"base64":STANDARD.encode(bytes),"mime":"image/png","name":"example.png","size":bytes.len()})
    }
    fn example() -> Value {
        json!({"version":1,"currentId":"chat-1","conversations":[{
            "id":"chat-1","title":"Markdown & images","createdAt":"2026-10-24T10:00:00Z",
            "updatedAt":"2026-10-24T10:01:00Z","settings":example_settings(),
            "draft":"## Draft\n\n**bold** and `code`\n","attachments":[png()],"messages":[
                {"role":"user","content":"Describe this image.\n","images":[png()],"mode":"chat"},
                {"role":"assistant","content":"# Answer\n\n```rust\nfn main() {}\n```",
                 "settings":example_settings(),"elapsed":0.125,"usage":{"prompt_tokens":12,
                 "completion_tokens":34,"total_tokens":46},"finish":"stop","model":"local-model"}
            ]
        }]})
    }

    #[test]
    fn missing_load_never_creates_files_or_directories() {
        let fixture = Fixture::new();
        let missing = fixture.0.join("not-created/nested");
        assert_eq!(load(&missing).unwrap(), empty_workspace());
        assert!(!fixture.0.join("not-created").exists());
        assert_eq!(load(&fixture.0).unwrap(), empty_workspace());
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 0);
        save(&missing, empty_workspace()).unwrap();
        assert_eq!(load(&missing).unwrap(), empty_workspace());
    }

    #[test]
    fn roundtrip_preserves_markdown_images_settings_and_optional_fields() {
        let fixture = Fixture::new();
        let workspace = example();
        save(&fixture.0, workspace.clone()).unwrap();
        assert_eq!(load(&fixture.0).unwrap(), workspace);
        let mut changed = workspace;
        changed["conversations"][0]["settings"]["seed"] = json!("");
        changed["conversations"][0]["messages"][1]["usage"] = json!({});
        changed["conversations"][0]["messages"][1]["elapsed"] = json!(0);
        let jpeg = b"\xff\xd8\xfffixture";
        changed["conversations"][0]["attachments"][0] = json!({"base64":STANDARD.encode(jpeg),
            "mime":"image/jpeg","name":"image.jpg","size":jpeg.len()});
        save(&fixture.0, changed.clone()).unwrap();
        assert_eq!(load(&fixture.0).unwrap(), changed);
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    fn reject_change(pointer: &str, value: Value) {
        let mut workspace = example();
        *workspace.pointer_mut(pointer).unwrap() = value;
        assert!(
            validate(&workspace).is_err(),
            "unexpectedly accepted {pointer}"
        );
    }

    #[test]
    fn rejects_stale_current_ids_duplicate_ids_and_wrong_root_schema() {
        reject_change("/currentId", json!("missing"));
        reject_change("/currentId", json!(""));
        reject_change("/currentId", json!(false));
        reject_change("/version", json!(2));
        reject_change("/version", json!(1.0));
        reject_change("/conversations/0/id", json!(""));
        reject_change("/conversations/0/id", json!("x".repeat(129)));
        reject_change("/conversations/0/title", json!("é".repeat(129)));
        reject_change("/conversations/0/createdAt", json!("x".repeat(65)));
        reject_change("/conversations/0/updatedAt", json!(null));
        let mut workspace = example();
        let duplicate = workspace["conversations"][0].clone();
        workspace["conversations"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        assert!(validate(&workspace).is_err());
        assert!(validate(&json!({"version":1,"conversations":[]})).is_err());
        assert!(validate(&json!([])).is_err());
    }

    #[test]
    fn rejects_unknown_fields_at_every_level_without_echoing_secrets() {
        for pointer in [
            "",
            "/conversations/0",
            "/conversations/0/settings",
            "/conversations/0/attachments/0",
            "/conversations/0/messages/0",
            "/conversations/0/messages/0/images/0",
            "/conversations/0/messages/1/settings",
            "/conversations/0/messages/1/usage",
        ] {
            let mut workspace = example();
            workspace
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("apiKey-SECRET".into(), json!("SECRET"));
            let error = validate(&workspace).unwrap_err();
            assert!(!error.contains("SECRET"));
            let fixture = Fixture::new();
            assert!(save(&fixture.0.join("absent"), workspace).is_err());
            assert!(!fixture.0.join("absent").exists());
        }
    }

    #[test]
    fn rejects_bad_settings_numbers_and_optional_nulls() {
        for (key, values) in [
            ("mode", vec![json!("tool"), json!(null)]),
            (
                "system",
                vec![json!(false), json!("x".repeat(MAX_TEXT + 1))],
            ),
            ("coding", vec![json!(0), json!("true")]),
            (
                "max_tokens",
                vec![json!(0), json!(-1), json!(1.5), json!(u32::MAX as u64 + 1)],
            ),
            (
                "temperature",
                vec![json!(-0.1), json!(2.1), json!("NaN"), Value::Null],
            ),
            ("top_p", vec![json!(0), json!(-0.1), json!(1.1)]),
            (
                "seed",
                vec![
                    json!("42"),
                    json!(-1),
                    json!(u32::MAX as u64 + 1),
                    Value::Null,
                ],
            ),
        ] {
            for value in values {
                reject_change(&format!("/conversations/0/settings/{key}"), value);
            }
        }
        for key in [
            "images", "mode", "settings", "elapsed", "usage", "finish", "model",
        ] {
            let mut workspace = example();
            workspace["conversations"][0]["messages"][0][key] = Value::Null;
            assert!(validate(&workspace).is_err(), "accepted null {key}");
        }
        reject_change("/conversations/0/messages/0/role", json!("system"));
        reject_change("/conversations/0/messages/1/elapsed", json!(-1));
        reject_change("/conversations/0/messages/1/usage/total_tokens", json!(1.5));
        reject_change("/conversations/0/messages/1/usage/prompt_tokens", json!(-1));
    }

    #[test]
    fn enforces_text_array_and_message_count_limits() {
        for pointer in [
            "/conversations/0/draft",
            "/conversations/0/messages/0/content",
        ] {
            reject_change(pointer, json!("x".repeat(MAX_TEXT + 1)));
        }
        reject_change("/conversations/0/attachments", json!(vec![png(); 9]));
        reject_change("/conversations/0/messages/0/images", json!(vec![png(); 9]));
        let minimal = json!({"role":"user","content":""});
        reject_change(
            "/conversations/0/messages",
            json!(vec![minimal.clone(); 1001]),
        );
        let mut workspace = example();
        workspace["conversations"][0]["messages"] = json!(vec![minimal; 1000]);
        let mut conversations = Vec::new();
        for n in 0..11 {
            let mut conversation = workspace["conversations"][0].clone();
            conversation["id"] = json!(format!("chat-{n}"));
            conversations.push(conversation);
        }
        workspace["conversations"] = json!(conversations);
        assert!(validate(&workspace).is_err());
        workspace["conversations"].as_array_mut().unwrap().pop();
        assert!(validate(&workspace).is_ok());
        reject_change("/conversations", json!(vec![Value::Null; 101]));
    }

    #[test]
    fn rejects_invalid_base64_mime_and_size_metadata() {
        for (key, values) in [
            (
                "base64",
                vec![
                    json!("%%%"),
                    json!("data:image/png;base64,iVBORw0KGgo="),
                    json!("iVBORw0KGgo=\n"),
                    json!("iVBORw0KGgp="),
                    json!("iVBORw0KGgo"),
                ],
            ),
            (
                "mime",
                vec![
                    json!("image/jpeg"),
                    json!("image/svg+xml"),
                    json!("IMAGE/PNG"),
                ],
            ),
            (
                "size",
                vec![json!(0), json!(-1), json!(1.5), json!(MAX_IMAGE + 1)],
            ),
        ] {
            for value in values {
                reject_change(&format!("/conversations/0/attachments/0/{key}"), value);
            }
        }
        let mut workspace = example();
        workspace["conversations"][0]["attachments"][0] = json!({"base64":STANDARD.encode(b"<script>"),
            "mime":"image/png","size":8,"name":"fake.png"});
        assert!(validate(&workspace).is_err());
        let mut bytes = vec![0; MAX_IMAGE];
        bytes[..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        workspace["conversations"][0]["attachments"][0] = json!({"base64":STANDARD.encode(&bytes),
            "mime":"image/png","size":bytes.len(),"name":"large.png"});
        assert!(validate(&workspace).is_ok());
        bytes.push(0);
        workspace["conversations"][0]["attachments"][0]["base64"] = json!(STANDARD.encode(&bytes));
        assert!(validate(&workspace).is_err());
    }

    #[test]
    fn malformed_unknown_or_oversized_saved_files_are_preserved() {
        let fixture = Fixture::new();
        for bytes in [
            b"{broken".to_vec(),
            b"{\"version\":2,\"currentId\":null,\"conversations\":[]}".to_vec(),
            b"{\"version\":1,\"currentId\":null,\"conversations\":[],\"apiKey\":\"secret\"}"
                .to_vec(),
        ] {
            fs::write(fixture.path(), &bytes).unwrap();
            assert!(load(&fixture.0).is_err());
            assert!(save(&fixture.0, empty_workspace()).is_err());
            assert_eq!(fs::read(fixture.path()).unwrap(), bytes);
        }
        let file = File::create(fixture.path()).unwrap();
        file.set_len(MAX_BYTES as u64 + 1).unwrap();
        drop(file);
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, empty_workspace()).is_err());
        assert_eq!(
            fs::metadata(fixture.path()).unwrap().len(),
            MAX_BYTES as u64 + 1
        );
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn invalid_incoming_and_oversized_serialization_preserve_existing_file() {
        let fixture = Fixture::new();
        save(&fixture.0, empty_workspace()).unwrap();
        let before = fs::read(fixture.path()).unwrap();
        assert!(save(&fixture.0, json!({"version":2})).is_err());
        assert_eq!(fs::read(fixture.path()).unwrap(), before);
        let mut workspace = example();
        // Escaping makes this exceed 64 MiB although each text field is <= 1 MiB.
        workspace["conversations"][0]["messages"] = json!(vec![
            json!({"role":"user","content":"\u{0000}".repeat(MAX_TEXT)});
            11
        ]);
        assert!(validate(&workspace).is_ok());
        assert!(save(&fixture.0, workspace).is_err());
        assert_eq!(fs::read(fixture.path()).unwrap(), before);
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn limited_writer_accepts_exact_limit_and_rejects_overflow() {
        let mut writer = LimitedWriter {
            bytes: vec![0; MAX_BYTES - 1],
        };
        assert_eq!(writer.write(b"x").unwrap(), 1);
        assert!(writer.write(b"y").is_err());
        assert_eq!(writer.bytes.len(), MAX_BYTES);
    }

    #[test]
    fn temporary_collisions_and_failures_never_delete_unowned_files() {
        let fixture = Fixture::new();
        let collision = fixture.0.join("collision.tmp");
        fs::write(&collision, b"keep").unwrap();
        assert!(create_temporary_at(collision.clone()).is_err());
        assert_eq!(fs::read(&collision).unwrap(), b"keep");
        let directory = fixture.0.join("directory.tmp");
        fs::create_dir(&directory).unwrap();
        assert!(create_temporary_at(directory.clone()).is_err());
        assert!(directory.is_dir());
        let (temporary, mut file) = create_temporary(&fixture.0).unwrap();
        let owned = temporary.path.clone();
        file.write_all(b"partial").unwrap();
        drop(file);
        // A failed rename leaves both the old destination and unrelated temps alone.
        assert!(fs::rename(&owned, &directory).is_err());
        drop(temporary);
        assert!(!owned.exists());
        assert!(directory.is_dir());
        assert_eq!(fs::read(&collision).unwrap(), b"keep");
    }

    #[test]
    fn concurrent_saves_always_leave_a_complete_valid_workspace() {
        let fixture = Fixture::new();
        std::thread::scope(|scope| {
            for n in 0..8 {
                let directory = &fixture.0;
                scope.spawn(move || {
                    let mut workspace = example();
                    workspace["conversations"][0]["draft"] = json!(format!("draft {n}"));
                    save(directory, workspace).unwrap();
                    assert!(validate(&load(directory).unwrap()).is_ok());
                });
            }
        });
        assert!(validate(&load(&fixture.0).unwrap()).is_ok());
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn private_permissions_apply_on_creation_and_replacement() {
        use std::os::unix::fs::PermissionsExt;
        let fixture = Fixture::new();
        let (temporary, file) = create_temporary(&fixture.0).unwrap();
        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        drop(file);
        drop(temporary);
        save(&fixture.0, example()).unwrap();
        assert_eq!(
            fs::metadata(fixture.path()).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::set_permissions(fixture.path(), fs::Permissions::from_mode(0o644)).unwrap();
        save(&fixture.0, empty_workspace()).unwrap();
        assert_eq!(
            fs::metadata(fixture.path()).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_directories_and_fifos_are_never_read_or_replaced() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let target = fixture.0.join("outside.json");
        fs::write(&target, b"keep").unwrap();
        symlink(&target, fixture.path()).unwrap();
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, empty_workspace()).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"keep");
        assert!(
            fs::symlink_metadata(fixture.path())
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(open_readonly(&fixture.path()).is_err());
        fs::remove_file(fixture.path()).unwrap();
        fs::create_dir(fixture.path()).unwrap();
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, empty_workspace()).is_err());
        fs::remove_dir(fixture.path()).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(fixture.path())
            .status()
            .unwrap();
        assert!(status.success());
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, empty_workspace()).is_err());
        // Even a raced FIFO open is nonblocking; the caller checks its file type.
        let file = open_readonly(&fixture.path()).unwrap();
        assert!(!file.metadata().unwrap().file_type().is_file());
        drop(file);
        fs::remove_file(fixture.path()).unwrap();
        symlink(fixture.0.join("missing-target"), fixture.path()).unwrap();
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, empty_workspace()).is_err());
    }
}
