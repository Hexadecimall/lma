//! lma-owned aliases. `delete` removes an alias only; high-level model deletion
//! (including managed, Ollama and LM Studio files) belongs to `models::delete`.
use super::modelfile::Definition;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions, TryLockError},
    io::{Read, Write},
    path::{Path, PathBuf},
};
const LIMIT: u64 = 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub path: PathBuf,
    pub projector: Option<PathBuf>,
}

// Definitions live in the same atomic document as paths. Keep Entry unchanged
// for discovery, sharing checks and deletion callers; do not use serde flatten
// with Entry's deny_unknown_fields, which would reject definition metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredEntry {
    path: PathBuf,
    projector: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    definition: Option<Definition>,
}
impl StoredEntry {
    fn entry(&self) -> Entry {
        Entry {
            path: self.path.clone(),
            projector: self.projector.clone(),
        }
    }
    fn new(entry: Entry, definition: Option<Definition>) -> Self {
        Self {
            path: entry.path,
            projector: entry.projector,
            definition,
        }
    }
}

pub fn name(input: &str) -> Result<String, String> {
    if input.len() > 512
        || input.starts_with("ollama:")
        || Path::new(input).is_absolute()
        || input.starts_with('.')
    {
        return Err("use a model name, not a filesystem path or scheme".into());
    }
    super::Reference::parse(input).map(|reference| reference.name())
}
fn location(root: &Path) -> PathBuf {
    root.join(".registry-v1.json")
}
fn lock_location(root: &Path) -> PathBuf {
    root.join(".registry-v1.lock")
}

fn regular_lock(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return false; // FILE_ATTRIBUTE_REPARSE_POINT
        }
    }
    metadata.is_file() && !metadata.file_type().is_symlink()
}

// Parent directories are trusted, as in local_file. Unlike model assets, the
// lock must never follow symlinks. Keep its inode separate from atomic JSON
// replacements, and never unlink it: other processes may still hold it open.
// File::try_lock requires Rust 1.89.0 and excludes independent handles even
// within one process. Closing the returned File releases the lock on all exits.
fn mutation_lock(root: &Path) -> Result<File, String> {
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let path = lock_location(root);
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800); // O_NOFOLLOW | O_NONBLOCK
    }
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4); // O_NOFOLLOW | O_NONBLOCK
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let invalid = || {
        format!(
            "{} must be a regular, non-symlink lock file",
            path.display()
        )
    };
    let file = match options.create_new(true).open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let before = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
            if !regular_lock(&before) {
                return Err(invalid());
            }
            options
                .create_new(false)
                .open(&path)
                .map_err(|e| format!("cannot open registry lock {}: {e}", path.display()))?
        }
        Err(e) => {
            return Err(format!(
                "cannot create registry lock {}: {e}",
                path.display()
            ));
        }
    };
    let opened = file.metadata().map_err(|e| e.to_string())?;
    let current = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !regular_lock(&opened) || !regular_lock(&current) {
        return Err(invalid());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if opened.dev() != current.dev() || opened.ino() != current.ino() {
            return Err("registry lock changed while opening it; retry".into());
        }
    }
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err("lma registry busy; retry the operation".into()),
        Err(TryLockError::Error(e)) => Err(format!("cannot lock registry {}: {e}", path.display())),
    }
}

fn read_full(root: &Path) -> Result<BTreeMap<String, StoredEntry>, String> {
    let path = location(root);
    match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(e.to_string()),
        Ok(meta) if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > LIMIT => {
            return Err("lma registry must be a regular file of at most 1 MiB".into());
        }
        _ => {}
    }
    let file = crate::local_file::open_regular(&path)?;
    let mut bytes = Vec::new();
    file.take(LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > LIMIT {
        return Err("lma registry exceeds 1 MiB".into());
    }
    struct Entries;
    impl<'de> serde::de::Visitor<'de> for Entries {
        type Value = BTreeMap<String, StoredEntry>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("unique lma registry labels")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut entries = BTreeMap::new();
            while let Some((label, entry)) = map.next_entry::<String, StoredEntry>()? {
                if entries.insert(label, entry).is_some() {
                    return Err(serde::de::Error::custom("duplicate lma registry label"));
                }
                if entries.len() > 4096 {
                    return Err(serde::de::Error::custom(
                        "lma registry entry limit exceeded",
                    ));
                }
            }
            Ok(entries)
        }
    }
    let mut decoder = serde_json::Deserializer::from_slice(&bytes);
    let entries = serde::de::Deserializer::deserialize_map(&mut decoder, Entries)
        .map_err(|e| format!("invalid lma registry: {e}"))?;
    decoder
        .end()
        .map_err(|e| format!("invalid lma registry: {e}"))?;
    if entries.len() > 4096 {
        return Err("lma registry entry limit exceeded".into());
    }
    for (label, entry) in &entries {
        if let Some(definition) = &entry.definition {
            definition
                .validate()
                .map_err(|e| format!("invalid definition for '{label}': {e}"))?;
        }
        if name(label)? != *label
            || !entry.path.is_absolute()
            || entry
                .projector
                .as_ref()
                .is_some_and(|path| !path.is_absolute())
        {
            return Err("invalid lma registry entry".into());
        }
    }
    Ok(entries)
}
pub fn read(root: &Path) -> Result<BTreeMap<String, Entry>, String> {
    Ok(read_full(root)?
        .into_iter()
        .map(|(label, stored)| (label, stored.entry()))
        .collect())
}

fn change_full(
    root: &Path,
    update: impl FnOnce(&mut BTreeMap<String, StoredEntry>) -> Result<(), String>,
) -> Result<(), String> {
    let _lock = mutation_lock(root)?;
    let mut entries = read_full(root)?;
    update(&mut entries)?;
    if entries.len() > 4096 {
        return Err("lma registry entry limit exceeded".into());
    }
    let bytes = serde_json::to_vec_pretty(&entries).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > LIMIT {
        return Err("lma registry exceeds 1 MiB".into());
    }
    let mut temporary = tempfile::NamedTempFile::new_in(root).map_err(|e| e.to_string())?;
    temporary
        .write_all(&bytes)
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    temporary
        .persist(location(root))
        .map_err(|e| e.to_string())?;
    Ok(())
}
fn change(
    root: &Path,
    update: impl FnOnce(&mut BTreeMap<String, Entry>) -> Result<(), String>,
) -> Result<(), String> {
    change_full(root, |stored| {
        let mut entries = stored
            .iter()
            .map(|(label, value)| (label.clone(), value.entry()))
            .collect();
        update(&mut entries)?;
        let previous = std::mem::take(stored);
        for (label, entry) in entries {
            let definition = previous
                .get(&label)
                .filter(|value| value.entry() == entry)
                .and_then(|value| value.definition.clone());
            stored.insert(label, StoredEntry::new(entry, definition));
        }
        Ok(())
    })
}

pub(crate) fn canonical_entry(entry: Entry) -> Result<Entry, String> {
    let entry = Entry {
        path: fs::canonicalize(&entry.path).map_err(|e| e.to_string())?,
        projector: entry
            .projector
            .map(fs::canonicalize)
            .transpose()
            .map_err(|e| e.to_string())?,
    };
    let mut magic = [0; 4];
    crate::local_file::open_regular(&entry.path)?
        .read_exact(&mut magic)
        .map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err("only GGUF model aliases can be registered; import safetensors first".into());
    }
    Ok(entry)
}

pub fn put(root: &Path, label: &str, entry: Entry) -> Result<String, String> {
    publish(root, label, entry, None)
}

/// Validate and publish both the alias and its definition with one atomic rename.
pub fn put_definition(
    root: &Path,
    label: &str,
    entry: Entry,
    definition: Definition,
) -> Result<String, String> {
    definition.validate()?;
    publish(root, label, entry, Some(definition))
}

fn publish(
    root: &Path,
    label: &str,
    entry: Entry,
    definition: Option<Definition>,
) -> Result<String, String> {
    let label = name(label)?;
    let entry = canonical_entry(entry)?;
    change_full(root, |entries| {
        if entries.contains_key(&label) {
            return Err(format!(
                "model '{label}' already exists; delete its lma alias first"
            ));
        }
        entries.insert(label.clone(), StoredEntry::new(entry, definition));
        Ok(())
    })?;
    Ok(label)
}

/// Missing aliases and legacy aliases return None; malformed registries are errors.
pub fn definition(root: &Path, label: &str) -> Result<Option<Definition>, String> {
    let label = name(label)?;
    Ok(read_full(root)?
        .remove(&label)
        .and_then(|entry| entry.definition))
}

/// Copy an alias and its definition without copying any model files.
pub fn copy(root: &Path, source: &str, destination: &str) -> Result<String, String> {
    let source = name(source)?;
    let destination = name(destination)?;
    change_full(root, |entries| {
        if entries.contains_key(&destination) {
            return Err(format!(
                "model '{destination}' already exists; delete its lma alias first"
            ));
        }
        let stored = entries
            .get(&source)
            .ok_or("source model is not a lma registry alias")?
            .clone();
        let entry = canonical_entry(stored.entry())?;
        entries.insert(
            destination.clone(),
            StoredEntry::new(entry, stored.definition),
        );
        Ok(())
    })?;
    Ok(destination)
}

/// Remove only a registry alias. Use `models::delete::run` for model-file deletion.
pub fn delete(root: &Path, label: &str) -> Result<(), String> {
    let label = name(label)?;
    change(root, |entries| {
        entries.remove(&label).map(|_| ()).ok_or_else(|| "model is not a lma registry alias; registry::delete removes aliases only; use models::delete::run for Ollama/LM Studio or managed-file deletion".into())
    })
}

/// Keep alias sharing checks and staged-file preparation under the same lock as
/// `put`/`delete`. A returned staging guard rolls files back if registry publication fails.
pub(super) fn remove_with<T>(
    root: &Path,
    label: &str,
    expected: &Entry,
    prepare: impl FnOnce(&BTreeMap<String, Entry>) -> Result<T, String>,
) -> Result<T, String> {
    let label = name(label)?;
    let mut prepared = None;
    change(root, |entries| {
        if entries.get(&label) != Some(expected) {
            return Err("registry alias changed before deletion; retry selection".into());
        }
        prepared = Some(prepare(entries)?);
        entries.remove(&label);
        Ok(())
    })?;
    prepared.ok_or_else(|| "registry deletion did not prepare an operation".into())
}

#[cfg(test)]
#[path = "../../tests/registry_process.rs"]
mod process_tests;

#[cfg(test)]
mod tests {
    use super::*;
    super::process_tests::registry_process_tests!();
    fn fixture() -> (tempfile::TempDir, PathBuf, Entry, Definition) {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("weights.gguf");
        fs::write(&model, b"GGUFfixture").unwrap();
        let root = dir.path().join("registry");
        let entry = Entry {
            path: fs::canonicalize(&model).unwrap(),
            projector: None,
        };
        let definition = super::super::modelfile::parse(
            "FROM fixture:latest\nSYSTEM persistent\nMESSAGE user hello\nPARAMETER temperature 0\nPARAMETER num_ctx 128"
        ).unwrap();
        (dir, root, entry, definition)
    }

    #[test]
    fn lock_is_stable_nonblocking_per_registry_and_released_on_errors() {
        let (_dir, root, entry, definition) = fixture();
        put_definition(&root, "defined", entry.clone(), definition.clone()).unwrap();
        let before = fs::read(location(&root)).unwrap();
        let guard = mutation_lock(&root).unwrap();
        let identity = guard.metadata().unwrap();
        assert!(mutation_lock(&root).unwrap_err().contains("registry busy"));
        assert_eq!(read(&root).unwrap()["defined:latest"], entry);
        assert_eq!(
            super::definition(&root, "defined").unwrap(),
            Some(definition)
        );
        assert_eq!(fs::read(location(&root)).unwrap(), before);
        let other = root.with_file_name("other-registry");
        drop(mutation_lock(&other).unwrap());
        // Lock acquisition precedes even parsing the registry document.
        fs::write(location(&root), b"invalid JSON").unwrap();
        assert!(
            delete(&root, "defined")
                .unwrap_err()
                .contains("registry busy")
        );
        fs::write(location(&root), &before).unwrap();
        drop(guard);
        assert!(change_full(&root, |_| Err("deliberate update failure".into())).is_err());
        assert!(
            remove_with(&root, "defined", &entry, |_| Err::<(), _>(
                "preparation failed".into()
            ))
            .is_err()
        );
        copy(&root, "defined", "copied").unwrap();
        delete(&root, "defined").unwrap();
        let current = fs::symlink_metadata(lock_location(&root)).unwrap();
        assert!(regular_lock(&current));
        assert_eq!(current.len(), 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                (identity.dev(), identity.ino()),
                (current.dev(), current.ino())
            );
        }
        #[cfg(not(unix))]
        let _ = identity;
    }

    #[test]
    fn read_only_lookups_never_create_a_directory_or_lock() {
        let (_dir, root, _entry, _definition) = fixture();
        assert!(read(&root).unwrap().is_empty());
        assert!(super::definition(&root, "missing").unwrap().is_none());
        assert!(!root.exists());
        fs::create_dir(&root).unwrap();
        fs::write(location(&root), b"{}").unwrap();
        assert!(read(&root).unwrap().is_empty());
        assert!(super::definition(&root, "missing").unwrap().is_none());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    }

    #[test]
    fn nonregular_lock_path_is_rejected_without_registry_publication() {
        let (_dir, root, entry, _definition) = fixture();
        fs::create_dir(&root).unwrap();
        fs::create_dir(lock_location(&root)).unwrap();
        assert!(put(&root, "new", entry).unwrap_err().contains("lock file"));
        assert!(!location(&root).exists());
        assert!(lock_location(&root).is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_locks_never_follow_or_modify_existing_or_missing_targets() {
        use std::os::unix::fs::symlink;
        for existing in [true, false] {
            let (dir, root, entry, definition) = fixture();
            put_definition(&root, "defined", entry.clone(), definition).unwrap();
            let before = fs::read(location(&root)).unwrap();
            let target = dir.path().join("lock-target");
            if existing {
                fs::write(&target, b"must not change").unwrap();
            }
            // No clients hold this fixture's lock while replacing it with the
            // malicious entry; production code must never remove the sidecar.
            fs::remove_file(lock_location(&root)).unwrap();
            symlink(&target, lock_location(&root)).unwrap();
            assert!(put(&root, "new", entry).unwrap_err().contains("lock file"));
            assert_eq!(fs::read(location(&root)).unwrap(), before);
            assert!(
                fs::symlink_metadata(lock_location(&root))
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            if existing {
                assert_eq!(fs::read(&target).unwrap(), b"must not change");
            } else {
                assert!(!target.exists());
            }
            assert!(read(&root).unwrap().contains_key("defined:latest"));
            assert!(super::definition(&root, "defined").unwrap().is_some());
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn fifo_lock_is_rejected_without_opening_or_blocking() {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        #[cfg(target_os = "macos")]
        type Mode = u16;
        #[cfg(target_os = "linux")]
        type Mode = u32;
        unsafe extern "C" {
            fn mkfifo(path: *const std::ffi::c_char, mode: Mode) -> i32;
        }
        let (_dir, root, entry, _definition) = fixture();
        fs::create_dir(&root).unwrap();
        let path = CString::new(lock_location(&root).as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(put(&root, "new", entry).unwrap_err().contains("lock file"));
        assert!(!location(&root).exists());
    }

    #[test]
    fn definitions_and_legacy_aliases_share_one_document_and_survive_changes() {
        let (_dir, root, entry, definition) = fixture();
        fs::create_dir(&root).unwrap();
        fs::write(
            location(&root),
            serde_json::to_vec(&BTreeMap::from([("legacy:latest", entry.clone())])).unwrap(),
        )
        .unwrap();
        assert_eq!(definition_lookup(&root, "legacy"), None);
        put_definition(&root, "defined", entry.clone(), definition.clone()).unwrap();
        put(&root, "ordinary", entry.clone()).unwrap();
        assert_eq!(
            definition_lookup(&root, "defined"),
            Some(definition.clone())
        );
        assert_eq!(read(&root).unwrap()["defined:latest"], entry);
        copy(&root, "defined", "copied").unwrap();
        assert_eq!(definition_lookup(&root, "copied"), Some(definition.clone()));
        delete(&root, "ordinary").unwrap();
        remove_with(&root, "legacy", &entry, |entries| {
            assert_eq!(entries["defined:latest"], entry);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition_lookup(&root, "defined"),
            Some(definition.clone())
        );
        delete(&root, "defined").unwrap();
        assert_eq!(definition_lookup(&root, "defined"), None);
        assert_eq!(definition_lookup(&root, "copied"), Some(definition));
        assert!(entry.path.exists());
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(location(&root)).unwrap()).unwrap();
        assert!(value["copied:latest"]["definition"].is_object());
        assert_eq!(
            fs::read_dir(&root).unwrap().count(),
            2,
            "only the registry document and stable lock sidecar"
        );
    }

    fn definition_lookup(root: &Path, label: &str) -> Option<Definition> {
        definition(root, label).unwrap()
    }

    #[test]
    fn failed_definition_publication_never_overwrites_alias_or_registry() {
        let (_dir, root, entry, definition) = fixture();
        put_definition(&root, "defined", entry.clone(), definition.clone()).unwrap();
        let before = fs::read(location(&root)).unwrap();
        let mut replacement = definition.clone();
        replacement.system = Some("replacement".into());
        assert!(put_definition(&root, "defined", entry.clone(), replacement).is_err());
        assert!(put(&root, "defined", entry.clone()).is_err());
        assert!(copy(&root, "defined", "defined").is_err());
        let mut invalid = definition.clone();
        invalid.parameters.num_ctx = Some(1);
        assert!(put_definition(&root, "invalid", entry.clone(), invalid).is_err());
        let mut oversized = definition;
        oversized.system = Some("\\\"".repeat(600_000));
        assert!(put_definition(&root, "oversized", entry, oversized).is_err());
        assert_eq!(fs::read(location(&root)).unwrap(), before);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
    }

    #[test]
    fn corrupt_unknown_duplicate_and_oversized_registries_are_never_replaced() {
        let (_dir, root, entry, definition) = fixture();
        fs::create_dir(&root).unwrap();
        let stored = serde_json::to_string(&StoredEntry::new(entry.clone(), None)).unwrap();
        let mut bad_definition = serde_json::to_value(&definition).unwrap();
        bad_definition["parameters"]["unknown"] = serde_json::json!(1);
        let cases = [
            b"not JSON".to_vec(),
            serde_json::to_vec(&serde_json::json!({"bad:latest":{"path":entry.path,"projector":null,"unknown":1}})).unwrap(),
            serde_json::to_vec(&serde_json::json!({"bad:latest":{"path":entry.path,"projector":null,"definition":bad_definition}})).unwrap(),
            format!("{{\"duplicate:latest\":{stored},\"duplicate:latest\":{stored}}}").into_bytes(),
            vec![b' '; LIMIT as usize + 1],
        ];
        for bytes in cases {
            fs::write(location(&root), &bytes).unwrap();
            assert!(read(&root).is_err());
            assert!(super::definition(&root, "new").is_err());
            assert!(put_definition(&root, "new", entry.clone(), definition.clone()).is_err());
            assert_eq!(fs::read(location(&root)).unwrap(), bytes);
        }
    }

    #[test]
    fn write_limits_and_gguf_validation_apply_before_atomic_publication() {
        let (dir, root, entry, definition) = fixture();
        put_definition(&root, "defined", entry.clone(), definition).unwrap();
        let before = fs::read(location(&root)).unwrap();
        assert!(
            change_full(&root, |entries| {
                for index in 0..4096 {
                    entries.insert(
                        format!("extra{index}:latest"),
                        StoredEntry::new(entry.clone(), None),
                    );
                }
                Ok(())
            })
            .is_err()
        );
        assert!(
            change_full(&root, |entries| {
                for index in 0..2000 {
                    entries.insert(
                        format!("{}-{index}:latest", "x".repeat(500)),
                        StoredEntry::new(entry.clone(), None),
                    );
                }
                Ok(())
            })
            .is_err()
        );
        let invalid = dir.path().join("invalid.gguf");
        fs::write(&invalid, b"nope").unwrap();
        assert!(
            put_definition(
                &root,
                "invalid",
                Entry {
                    path: invalid,
                    projector: None
                },
                super::super::modelfile::parse("FROM source").unwrap()
            )
            .is_err()
        );
        assert_eq!(fs::read(location(&root)).unwrap(), before);
        assert!(entry.path.exists());
    }

    #[test]
    fn aliases_are_atomic_bounded_and_never_delete_weights() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("weights.gguf");
        fs::write(&model, b"GGUFfixture").unwrap();
        let root = directory.path().join("store");
        let entry = Entry {
            path: model.clone(),
            projector: None,
        };
        assert_eq!(put(&root, "test", entry.clone()).unwrap(), "test:latest");
        assert!(put(&root, "test", entry).is_err());
        assert_eq!(
            read(&root).unwrap()["test:latest"].path,
            fs::canonicalize(&model).unwrap()
        );
        delete(&root, "test").unwrap();
        assert!(read(&root).unwrap().is_empty());
        assert!(model.exists());
        assert!(delete(&root, "test").is_err());
        for bad in ["../../oops", "/tmp/model", "ollama:model", "a\nb"] {
            assert!(name(bad).is_err());
        }
    }
}
