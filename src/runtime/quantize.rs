use std::ffi::CString;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_sys_2 as sys;

use crate::config::QuantizeConfig;

pub fn run(config: QuantizeConfig) -> Result<(), String> {
    let ftype = file_type(&config.kind)?;
    // Initialize lazily, after path validation, but keep the backend alive until
    // publication and temporary-file cleanup have also finished.
    let mut backend = None;
    with_output(&config, |input, output| {
        let backend = backend.insert(
            LlamaBackend::init()
                .map_err(|error| format!("could not initialize llama backend: {error}"))?,
        );
        quantize(backend, input, output, &config, ftype)
    })
}

fn file_type(kind: &str) -> Result<sys::llama_ftype, String> {
    Ok(match kind {
        "F32" => sys::LLAMA_FTYPE_ALL_F32,
        "F16" => sys::LLAMA_FTYPE_MOSTLY_F16,
        "BF16" => sys::LLAMA_FTYPE_MOSTLY_BF16,
        "Q4_0" => sys::LLAMA_FTYPE_MOSTLY_Q4_0,
        "Q4_1" => sys::LLAMA_FTYPE_MOSTLY_Q4_1,
        "Q5_0" => sys::LLAMA_FTYPE_MOSTLY_Q5_0,
        "Q5_1" => sys::LLAMA_FTYPE_MOSTLY_Q5_1,
        "Q8_0" => sys::LLAMA_FTYPE_MOSTLY_Q8_0,
        "Q2_K" => sys::LLAMA_FTYPE_MOSTLY_Q2_K,
        "Q3_K_S" => sys::LLAMA_FTYPE_MOSTLY_Q3_K_S,
        "Q3_K_M" => sys::LLAMA_FTYPE_MOSTLY_Q3_K_M,
        "Q3_K_L" => sys::LLAMA_FTYPE_MOSTLY_Q3_K_L,
        "Q4_K_S" => sys::LLAMA_FTYPE_MOSTLY_Q4_K_S,
        "Q4_K_M" => sys::LLAMA_FTYPE_MOSTLY_Q4_K_M,
        "Q5_K_S" => sys::LLAMA_FTYPE_MOSTLY_Q5_K_S,
        "Q5_K_M" => sys::LLAMA_FTYPE_MOSTLY_Q5_K_M,
        "Q6_K" => sys::LLAMA_FTYPE_MOSTLY_Q6_K,
        _ => return Err(format!("unsupported quantization type: {kind}")),
    })
}

fn parameters(
    config: &QuantizeConfig,
    ftype: sys::llama_ftype,
) -> sys::llama_model_quantize_params {
    // SAFETY: this takes no arguments and returns a fully initialized value.
    let mut params = unsafe { sys::llama_model_quantize_default_params() };
    params.ftype = ftype;
    // llama.cpp uses hardware concurrency for any nonpositive value.
    params.nthread = config.threads;
    params.pure_ = config.pure;
    params.quantize_output_tensor = !config.leave_output;
    params.allow_requantize = config.allow_requantize;
    params
}

fn quantize(
    _backend: &LlamaBackend,
    input: &Path,
    output: &Path,
    config: &QuantizeConfig,
    ftype: sys::llama_ftype,
) -> Result<(), String> {
    let input_c = path_cstring(input)?;
    let output_c = path_cstring(output)?;
    let params = parameters(config, ftype);
    // SAFETY: both strings and params live throughout the synchronous call. The
    // default optional pointers are null; no imatrix, overrides or splits are used.
    // The borrowed backend remains initialized until the call has returned.
    let status = unsafe { sys::llama_model_quantize(input_c.as_ptr(), output_c.as_ptr(), &params) };
    if status != 0 {
        return Err(format!(
            "quantization of '{}' to {} failed (llama.cpp status {status}); see llama.cpp diagnostics above",
            input.display(),
            config.kind
        ));
    }
    Ok(())
}

fn path_cstring(path: &Path) -> Result<CString, String> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: '{}'", path.display()))?
        .as_bytes();
    CString::new(bytes).map_err(|_| format!("path contains a NUL byte: '{}'", path.display()))
}

struct Paths {
    input: PathBuf,
    output: PathBuf,
    // Keep the input inode alive, including while checking hard-link aliases.
    input_file: File,
}

impl Paths {
    fn prepare(config: &QuantizeConfig) -> Result<Self, String> {
        path_cstring(&config.input)?;
        path_cstring(&config.output)?;
        let metadata = fs::metadata(&config.input).map_err(|error| {
            format!("cannot access input '{}': {error}", config.input.display())
        })?;
        if !metadata.is_file() {
            return Err(format!(
                "input is not a regular file: '{}'",
                config.input.display()
            ));
        }
        let input = fs::canonicalize(&config.input).map_err(|error| {
            format!("cannot resolve input '{}': {error}", config.input.display())
        })?;
        let input_file = File::open(&input)
            .map_err(|error| format!("cannot open input '{}': {error}", input.display()))?;
        let name = config.output.file_name().ok_or_else(|| {
            format!(
                "output must name a regular file: '{}'",
                config.output.display()
            )
        })?;
        let parent = config
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // Resolve the parent, not the final component: forced publication must
        // replace a symlink itself rather than write through to its target.
        let parent = fs::canonicalize(parent).map_err(|error| {
            format!(
                "cannot resolve output directory '{}': {error}",
                parent.display()
            )
        })?;
        if !parent.is_dir() {
            return Err(format!(
                "output parent is not a directory: '{}'",
                parent.display()
            ));
        }
        let paths = Self {
            input,
            output: parent.join(name),
            input_file,
        };
        paths.check_output(config.force)?;
        Ok(paths)
    }

    fn check_output(&self, force: bool) -> Result<(), String> {
        let input_metadata = self
            .input_file
            .metadata()
            .map_err(|error| format!("cannot inspect input: {error}"))?;
        if self.input == self.output {
            return Err("input and output refer to the same file (even with -f)".into());
        }
        let entry = match fs::symlink_metadata(&self.output) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "cannot inspect output '{}': {error}",
                    self.output.display()
                ));
            }
        };
        match fs::metadata(&self.output) {
            Ok(metadata) => {
                if same_identity(&input_metadata, &metadata)
                    || fs::canonicalize(&self.output).map_err(|error| {
                        format!("cannot resolve output '{}': {error}", self.output.display())
                    })? == self.input
                {
                    return Err("input and output refer to the same file (even with -f)".into());
                }
                if !metadata.is_file() {
                    return Err(format!(
                        "output is not a regular file: '{}'",
                        self.output.display()
                    ));
                }
                // Without portable stable file IDs, fail closed rather than
                // potentially accepting a hard link to the input on other OSes.
                #[cfg(not(unix))]
                if force {
                    return Err("safe forced replacement of an existing output requires Unix file identity checks".into());
                }
            }
            Err(error)
                if entry.file_type().is_symlink() && error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "cannot inspect output '{}': {error}",
                    self.output.display()
                ));
            }
        }
        if !force {
            return Err(format!(
                "output already exists: '{}' (use -f to replace it; NSS: `lma nss quantize INPUT OUTPUT --force`)",
                self.output.display()
            ));
        }
        Ok(())
    }
}

#[cfg(unix)]
fn same_identity(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_identity(_left: &Metadata, _right: &Metadata) -> bool {
    false
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempOutput {
    path: PathBuf,
    file: File,
}

impl TempOutput {
    fn reserve(parent: &Path) -> Result<Self, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("cannot generate temporary filename: {error}"))?
            .as_nanos();
        for _ in 0..128 {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".lma-quantize-{}-{timestamp}-{counter}.gguf",
                std::process::id()
            ));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => return Ok(Self { path, file }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "cannot reserve temporary output in '{}': {error}",
                        parent.display()
                    ));
                }
            }
        }
        Err(format!(
            "could not reserve a unique temporary output in '{}'",
            parent.display()
        ))
    }

    fn is_owned(&self) -> bool {
        let Ok(entry) = fs::symlink_metadata(&self.path) else {
            return false;
        };
        let Ok(metadata) = self.file.metadata() else {
            return false;
        };
        #[cfg(unix)]
        {
            entry.is_file() && same_identity(&entry, &metadata)
        }
        #[cfg(not(unix))]
        {
            entry.is_file() && metadata.is_file()
        }
    }

    fn publish(&self, paths: &Paths, force: bool) -> Result<(), String> {
        if !self.is_owned() {
            return Err("temporary quantization output was replaced or removed".into());
        }
        if self
            .file
            .metadata()
            .map_err(|error| format!("cannot inspect temporary output: {error}"))?
            .len()
            == 0
        {
            return Err("quantization produced an empty output".into());
        }
        self.file
            .sync_all()
            .map_err(|error| format!("cannot flush quantized output: {error}"))?;
        paths.check_output(force)?;
        let result = if force {
            fs::rename(&self.path, &paths.output)
        } else {
            // Unlike check-then-rename, link creation atomically refuses ANY
            // existing destination, including a dangling symlink racing us.
            // Do not fall back to copying: that would expose partial output.
            fs::hard_link(&self.path, &paths.output)
        };
        result.map_err(|error| {
            format!(
                "cannot publish quantized output '{}': {error}",
                paths.output.display()
            )
        })
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        if self.is_owned()
            && let Err(error) = fs::remove_file(&self.path)
        {
            eprintln!(
                "lma: could not remove temporary output '{}': {error}",
                self.path.display()
            );
        }
    }
}

fn with_output(
    config: &QuantizeConfig,
    convert: impl FnOnce(&Path, &Path) -> Result<(), String>,
) -> Result<(), String> {
    let paths = Paths::prepare(config)?;
    let temp = TempOutput::reserve(paths.output.parent().expect("resolved output has a parent"))?;
    // llama.cpp reopens the reserved filename. As with other filename-based
    // native APIs, the containing directory must not be attacker-controlled.
    convert(&paths.input, &temp.path)?;
    temp.publish(&paths, config.force)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let temp = TempOutput::reserve(&std::env::temp_dir()).unwrap();
            let dir = temp.path.with_extension("tests");
            fs::create_dir(&dir).unwrap();
            Self(dir)
        }

        fn config(&self) -> QuantizeConfig {
            let input = self.0.join("input.gguf");
            fs::write(&input, b"original").unwrap();
            QuantizeConfig {
                input,
                output: self.0.join("output.gguf"),
                kind: "Q4_K_M".into(),
                threads: 2,
                pure: false,
                leave_output: false,
                allow_requantize: false,
                force: false,
            }
        }

        fn assert_no_temps(&self) {
            assert!(fs::read_dir(&self.0).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".lma-quantize-")
            }));
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn supported_types_match_pinned_header() {
        for (kind, value) in [
            ("F32", 0),
            ("F16", 1),
            ("BF16", 32),
            ("Q4_0", 2),
            ("Q4_1", 3),
            ("Q5_0", 8),
            ("Q5_1", 9),
            ("Q8_0", 7),
            ("Q2_K", 10),
            ("Q3_K_S", 11),
            ("Q3_K_M", 12),
            ("Q3_K_L", 13),
            ("Q4_K_S", 14),
            ("Q4_K_M", 15),
            ("Q5_K_S", 16),
            ("Q5_K_M", 17),
            ("Q6_K", 18),
        ] {
            assert_eq!(file_type(kind).unwrap(), value);
        }
        for kind in ["", "q4_k_m", "COPY", "IQ2_XXS", "Q4_K"] {
            assert!(file_type(kind).is_err());
        }
    }

    #[test]
    fn options_preserve_basic_defaults() {
        let fixture = Fixture::new();
        let mut config = fixture.config();
        config.threads = -1;
        config.pure = true;
        config.leave_output = true;
        config.allow_requantize = true;
        let params = parameters(&config, file_type(&config.kind).unwrap());
        assert_eq!(params.ftype, sys::LLAMA_FTYPE_MOSTLY_Q4_K_M);
        assert_eq!(params.nthread, -1);
        assert!(params.pure_ && params.allow_requantize);
        assert!(!params.quantize_output_tensor);
        assert!(!params.only_copy && !params.keep_split && !params.dry_run);
        assert!(
            params.imatrix.is_null()
                && params.kv_overrides.is_null()
                && params.tt_overrides.is_null()
        );
        assert!(params.prune_layers.is_null());
    }

    #[test]
    fn same_path_is_rejected_even_with_force() {
        let fixture = Fixture::new();
        let mut config = fixture.config();
        config.output = fixture.0.join(".").join("input.gguf");
        config.force = true;
        assert!(Paths::prepare(&config).err().unwrap().contains("same file"));
        assert_eq!(fs::read(config.input).unwrap(), b"original");
    }

    #[cfg(unix)]
    #[test]
    fn hardlinks_and_symlinks_to_input_are_rejected() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let mut config = fixture.config();
        config.force = true;
        fs::hard_link(&config.input, &config.output).unwrap();
        assert!(Paths::prepare(&config).err().unwrap().contains("same file"));
        fs::remove_file(&config.output).unwrap();
        symlink(&config.input, &config.output).unwrap();
        assert!(Paths::prepare(&config).err().unwrap().contains("same file"));
    }

    #[test]
    fn missing_input_and_directories_are_rejected() {
        let fixture = Fixture::new();
        let mut config = fixture.config();
        fs::create_dir(&config.output).unwrap();
        config.force = true;
        assert!(
            Paths::prepare(&config)
                .err()
                .unwrap()
                .contains("not a regular file")
        );
        config.input = fixture.0.clone();
        assert!(
            Paths::prepare(&config)
                .err()
                .unwrap()
                .contains("input is not a regular file")
        );
        config.input = fixture.0.join("missing");
        assert!(
            Paths::prepare(&config)
                .err()
                .unwrap()
                .contains("cannot access input")
        );
    }

    #[test]
    fn existing_output_is_not_touched() {
        let fixture = Fixture::new();
        let config = fixture.config();
        fs::write(&config.output, b"existing").unwrap();
        assert!(
            with_output(&config, |_, _| panic!("must not quantize"))
                .unwrap_err()
                .contains("already exists")
        );
        assert_eq!(fs::read(&config.output).unwrap(), b"existing");
        fixture.assert_no_temps();
    }

    #[test]
    fn success_publishes_and_failure_cleans_partial_output() {
        let fixture = Fixture::new();
        let config = fixture.config();
        assert!(
            with_output(&config, |_, temp| {
                fs::write(temp, b"partial").unwrap();
                Err("conversion failed".into())
            })
            .is_err()
        );
        assert!(!config.output.exists());
        fixture.assert_no_temps();
        with_output(&config, |_, temp| {
            assert!(!config.output.exists());
            fs::write(temp, b"complete").unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(&config.output).unwrap(), b"complete");
        assert_eq!(fs::read(&config.input).unwrap(), b"original");
        fixture.assert_no_temps();
    }

    #[test]
    fn publication_does_not_clobber_a_racing_destination() {
        let fixture = Fixture::new();
        let config = fixture.config();
        let paths = Paths::prepare(&config).unwrap();
        let temp = TempOutput::reserve(&fixture.0).unwrap();
        fs::write(&temp.path, b"complete").unwrap();
        fs::write(&config.output, b"racer").unwrap();
        assert!(temp.publish(&paths, false).is_err());
        assert_eq!(fs::read(&config.output).unwrap(), b"racer");
    }

    #[cfg(unix)]
    #[test]
    fn force_replaces_only_after_success_and_rechecks_input_aliases() {
        let fixture = Fixture::new();
        let mut config = fixture.config();
        config.force = true;
        fs::write(&config.output, b"existing").unwrap();
        assert!(
            with_output(&config, |_, temp| {
                fs::write(temp, b"partial").unwrap();
                Err("failure".into())
            })
            .is_err()
        );
        assert_eq!(fs::read(&config.output).unwrap(), b"existing");
        with_output(&config, |_, temp| {
            fs::write(temp, b"complete").unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(&config.output).unwrap(), b"complete");
        assert!(
            with_output(&config, |input, temp| {
                fs::write(temp, b"replacement").unwrap();
                fs::remove_file(&config.output).unwrap();
                fs::hard_link(input, &config.output).unwrap();
                Ok(())
            })
            .unwrap_err()
            .contains("same file")
        );
        assert_eq!(fs::read(&config.input).unwrap(), b"original");
        fixture.assert_no_temps();
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_is_an_existing_output_and_force_does_not_follow_it() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let mut config = fixture.config();
        let target = fixture.0.join("missing-target");
        symlink(&target, &config.output).unwrap();
        assert!(
            Paths::prepare(&config)
                .err()
                .unwrap()
                .contains("already exists")
        );
        config.force = true;
        with_output(&config, |_, temp| {
            fs::write(temp, b"complete").unwrap();
            Ok(())
        })
        .unwrap();
        assert!(!target.exists());
        assert!(
            !fs::symlink_metadata(&config.output)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        fixture.assert_no_temps();
    }

    #[test]
    fn empty_output_is_not_published() {
        let fixture = Fixture::new();
        let config = fixture.config();
        assert!(
            with_output(&config, |_, _| Ok(()))
                .unwrap_err()
                .contains("empty output")
        );
        assert!(!config.output.exists());
        fixture.assert_no_temps();
    }

    #[cfg(unix)]
    #[test]
    fn unix_paths_preserve_bytes_and_reject_nul() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let bytes = b"non-utf8-\xff.gguf";
        assert_eq!(
            path_cstring(Path::new(OsStr::from_bytes(bytes)))
                .unwrap()
                .as_bytes(),
            bytes
        );
        assert!(path_cstring(Path::new(OsStr::from_bytes(b"bad\0path"))).is_err());
    }
}
