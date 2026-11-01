//! Self-update only the installed CLI. Release checksums authenticate transport/content,
//! not the publisher: SHA256SUMS is not an independent signature.
use reqwest::{
    StatusCode, Url,
    blocking::{Client, Response},
    header,
    redirect::Policy,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File, Metadata, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    time::Duration,
};
use tempfile::NamedTempFile;

const API: &str = "https://api.github.com/repos/Hexadecimall/lma";
const MAX_METADATA: u64 = 8 * 1024 * 1024;
const MAX_CHECKSUMS: u64 = 1024 * 1024;
const MAX_BINARY: u64 = 1024 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Version(u64, u64, u64);

impl Version {
    fn parse(input: &str) -> Result<Self> {
        let value = input.strip_prefix('v').unwrap_or(input);
        let numbers: Vec<_> = value.split('.').collect();
        if numbers.len() != 3 {
            return Err("version must be a stable MAJOR.MINOR.PATCH (optional leading v)".into());
        }
        let mut parsed = [0; 3];
        for (index, number) in numbers.iter().enumerate() {
            if number.is_empty()
                || !number.bytes().all(|b| b.is_ascii_digit())
                || (number.len() > 1 && number.starts_with('0'))
            {
                return Err(
                    "version must contain three canonical integers; prereleases are not supported"
                        .into(),
                );
            }
            parsed[index] = number.parse().map_err(|_| "version integer is too large")?;
        }
        Ok(Self(parsed[0], parsed[1], parsed[2]))
    }

    fn tag(self) -> String {
        format!("v{}.{}.{}", self.0, self.1, self.2)
    }
}

#[derive(Clone, Debug)]
struct Platform {
    target: String,
    backend: &'static str,
    os: String,
    arch: String,
}

impl Platform {
    fn ensure_runtime_requirements(&self) -> Result<()> {
        // Keep these minimums in sync with the official release workflow. A
        // matching architecture alone does not make a newer OS ABI executable.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            unsafe extern "C" {
                fn gnu_get_libc_version() -> *const std::ffi::c_char;
            }
            let pointer = unsafe { gnu_get_libc_version() };
            if pointer.is_null() {
                return Err("cannot determine glibc version; reinstall manually".into());
            }
            let version = unsafe { std::ffi::CStr::from_ptr(pointer) }
                .to_str()
                .map_err(|_| "invalid glibc version")?;
            if !version_at_least(version, 2, 39) {
                return Err("official GNU/Linux release binaries require glibc 2.39 or newer; keep this executable and use the source installer or owning package manager".into());
            }
        }
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sysctlbyname(
                    name: *const std::ffi::c_char,
                    old: *mut std::ffi::c_void,
                    len: *mut usize,
                    new: *mut std::ffi::c_void,
                    new_len: usize,
                ) -> std::ffi::c_int;
            }
            let mut bytes = [0u8; 128];
            let mut len = bytes.len();
            let result = unsafe {
                sysctlbyname(
                    c"kern.osrelease".as_ptr(),
                    bytes.as_mut_ptr().cast(),
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if result != 0 || len == 0 || len > bytes.len() {
                return Err("cannot determine macOS runtime version; reinstall manually".into());
            }
            let version = std::str::from_utf8(&bytes[..len])
                .map_err(|_| "invalid Darwin version")?
                .trim_end_matches('\0');
            if !version_at_least(version, 24, 0) {
                return Err("official Apple Silicon release binaries require macOS 15 or newer; keep this executable and use a compatible source build".into());
            }
        }
        Ok(())
    }

    fn compiled() -> Result<Self> {
        if !cfg!(target_endian = "little")
            || !cfg!(target_abi = "")
            || (std::env::consts::ARCH != "x86" && !cfg!(target_pointer_width = "64"))
        {
            return Err(
                "self-update does not support this compiled endianness/ABI; reinstall manually"
                    .into(),
            );
        }
        let environment = if cfg!(target_env = "musl") {
            "musl"
        } else if cfg!(target_env = "gnu") {
            "gnu"
        } else if cfg!(target_env = "msvc") {
            "msvc"
        } else {
            ""
        };
        Self::select(
            std::env::consts::OS,
            std::env::consts::ARCH,
            environment,
            cfg!(feature = "metal"),
            cfg!(feature = "cuda"),
        )
    }

    fn select(os: &str, arch: &str, environment: &str, metal: bool, cuda: bool) -> Result<Self> {
        let metal = metal || (os == "macos" && arch == "aarch64");
        if metal && cuda {
            return Err("no release asset can preserve both compiled Metal and CUDA backends; reinstall this build manually".into());
        }
        let backend = if metal {
            "metal"
        } else if cuda {
            "cuda"
        } else {
            "cpu"
        };
        let target = match (os, arch, environment, backend) {
            ("macos", "aarch64", _, "metal") => "aarch64-apple-darwin".into(),
            ("linux", "x86_64" | "aarch64", "gnu" | "musl", "cpu" | "cuda") => {
                format!("{arch}-unknown-linux-{environment}")
            }
            ("windows", "x86_64" | "aarch64" | "x86", "msvc" | "gnu", "cpu" | "cuda") => {
                let arch = if arch == "x86" { "i686" } else { arch };
                format!("{arch}-pc-windows-{environment}")
            }
            ("freebsd", "x86_64" | "aarch64", _, "cpu") => format!("{arch}-unknown-freebsd"),
            _ => {
                return Err(format!(
                    "self-update does not support this compiled target/backend ({os}/{arch}/{environment}/{backend}); reinstall manually"
                ));
            }
        };
        Ok(Self {
            target,
            backend,
            os: os.into(),
            arch: arch.into(),
        })
    }

    fn asset_name(&self, version: Version) -> String {
        format!(
            "lma-{}-{}-{}{}",
            version.tag(),
            self.target,
            self.backend,
            if self.os == "windows" { ".exe" } else { "" }
        )
    }
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    draft: bool,
    prerelease: bool,
    assets: Vec<Asset>,
}

#[derive(Debug, Deserialize)]
struct Asset {
    id: u64,
    name: String,
    state: String,
    size: u64,
    #[serde(default)]
    digest: Option<String>,
}

impl Release {
    fn version(&self, requested: Option<Version>) -> Result<Version> {
        if self.draft || self.prerelease {
            return Err(
                "release is a draft or prerelease; only stable published releases are accepted"
                    .into(),
            );
        }
        let version = Version::parse(&self.tag_name)?;
        if self.tag_name != version.tag() || requested.is_some_and(|v| v != version) {
            return Err(
                "release tag does not match the requested canonical vMAJOR.MINOR.PATCH".into(),
            );
        }
        Ok(version)
    }

    fn assets(&self, name: &str) -> Result<(&Asset, &Asset)> {
        let mut names = HashSet::new();
        let mut ids = HashSet::new();
        for asset in &self.assets {
            if !names.insert(&asset.name) || asset.id == 0 || !ids.insert(asset.id) {
                return Err(
                    "release metadata contains duplicate asset names/IDs or an invalid asset ID"
                        .into(),
                );
            }
        }
        let binary = self.assets.iter().find(|a| a.name == name)
            .ok_or_else(|| format!("release has no exact asset {name}; no backend fallback is allowed; reinstall manually or wait for a matching release"))?;
        let checksums = self
            .assets
            .iter()
            .find(|a| a.name == "SHA256SUMS")
            .ok_or("release has no SHA256SUMS; refusing an unverifiable update")?;
        binary.validate(MAX_BINARY)?;
        checksums.validate(MAX_CHECKSUMS)?;
        Ok((binary, checksums))
    }
}

impl Asset {
    fn validate(&self, maximum: u64) -> Result<()> {
        if self.id == 0 || self.state != "uploaded" || self.size == 0 || self.size > maximum {
            return Err("release asset must be uploaded, have a positive ID/size, and be within the download size limit".into());
        }
        self.sha256()?;
        Ok(())
    }

    fn sha256(&self) -> Result<Option<[u8; 32]>> {
        self.digest
            .as_deref()
            .map(|digest| {
                parse_hex(
                    digest
                        .strip_prefix("sha256:")
                        .ok_or("unsupported GitHub asset digest; expected sha256")?,
                )
            })
            .transpose()
    }
}

fn parse_hex(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid SHA-256 digest".into());
    }
    let mut digest = [0; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| "invalid SHA-256 digest")?;
    }
    Ok(digest)
}

fn checksum_entry(bytes: &[u8], wanted: &str) -> Result<[u8; 32]> {
    let text = std::str::from_utf8(bytes).map_err(|_| "SHA256SUMS is not UTF-8")?;
    let mut names = HashSet::new();
    let mut selected = None;
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        // Standard sha256sum text/binary records only, never escaped paths or comments.
        let bytes = line.as_bytes();
        if bytes.len() < 67 || bytes[64] != b' ' || !matches!(bytes[65], b' ' | b'*') {
            return Err("malformed SHA256SUMS entry (expected sha256sum format)".into());
        }
        let digest = parse_hex(std::str::from_utf8(&bytes[..64]).map_err(|_| "invalid checksum")?)?;
        let name = &line[66..];
        if name.is_empty()
            || name == "."
            || name == ".."
            || name
                .bytes()
                .any(|b| !b.is_ascii_alphanumeric() && !matches!(b, b'.' | b'_' | b'-'))
            || !names.insert(name)
        {
            return Err("SHA256SUMS contains an unsafe or duplicate filename".into());
        }
        if name == wanted {
            selected = Some(digest);
        }
    }
    selected
        .ok_or_else(|| "SHA256SUMS must contain exactly one entry for the selected binary".into())
}

/// A reader is injected here so all streaming/error paths can be tested offline.
fn copy_verified<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    expected_size: u64,
    maximum: u64,
    checksums_digest: Option<[u8; 32]>,
    api_digest: Option<[u8; 32]>,
) -> Result<[u8; 32]> {
    if expected_size == 0 || expected_size > maximum {
        return Err("download size is outside the permitted range".into());
    }
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0; 64 * 1024];
    loop {
        // Read at most one excess byte; never consume an unbounded response.
        let allowance = (expected_size - count + 1).min(buffer.len() as u64) as usize;
        let read = reader
            .read(&mut buffer[..allowance])
            .map_err(|_| "download stream failed; original executable was not replaced")?;
        if read == 0 {
            break;
        }
        count += read as u64;
        if count > expected_size {
            return Err(
                "download exceeds its declared size; original executable was not replaced".into(),
            );
        }
        hash.update(&buffer[..read]);
        writer
            .write_all(&buffer[..read])
            .map_err(|_| "cannot write update temporary file")?;
    }
    if count != expected_size {
        return Err("download is truncated (size mismatch)".into());
    }
    let digest: [u8; 32] = hash.finalize().into();
    if checksums_digest.is_some_and(|expected| expected != digest)
        || api_digest.is_some_and(|expected| expected != digest)
    {
        return Err("SHA-256 mismatch; original executable was not replaced".into());
    }
    Ok(digest)
}

struct GitHub {
    client: Client,
    token: Option<header::HeaderValue>,
}

fn trusted_url(url: &Url) -> bool {
    url.scheme() == "https"
        && url.port_or_known_default() == Some(443)
        && url.username().is_empty()
        && url.password().is_none()
        && url.fragment().is_none()
        && matches!(
            url.host_str(),
            Some(
                "api.github.com"
                    | "github.com"
                    | "objects.githubusercontent.com"
                    | "release-assets.githubusercontent.com"
            )
        )
}

fn api_url(url: &Url) -> bool {
    trusted_url(url)
        && url.host_str() == Some("api.github.com")
        && url.path().starts_with("/repos/Hexadecimall/lma/releases/")
        && url.query().is_none()
}

#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", target_env = "gnu"),
    test
))]
fn version_at_least(value: &str, major: u32, minor: u32) -> bool {
    let mut parts = value.split('.');
    match (
        parts.next().and_then(|s| s.parse::<u32>().ok()),
        parts.next().and_then(|s| s.parse::<u32>().ok()),
    ) {
        (Some(found_major), Some(found_minor)) => (found_major, found_minor) >= (major, minor),
        _ => false,
    }
}

fn optional_token() -> Result<Option<header::HeaderValue>> {
    for key in ["GH_TOKEN", "GITHUB_TOKEN"] {
        match std::env::var(key) {
            Ok(value) if !value.trim().is_empty() => {
                let mut value = header::HeaderValue::from_str(&format!("Bearer {}", value.trim()))
                    .map_err(|_| "GitHub token contains invalid header characters; fix GH_TOKEN/GITHUB_TOKEN")?;
                value.set_sensitive(true);
                return Ok(Some(value));
            }
            Ok(_) | Err(std::env::VarError::NotPresent) => {}
            Err(_) => {
                return Err("GitHub token is not valid Unicode; fix GH_TOKEN/GITHUB_TOKEN".into());
            }
        }
    }
    Ok(None)
}

impl GitHub {
    fn new() -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .https_only(true)
                .redirect(Policy::none())
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(300))
                .user_agent(concat!("lma-updater/", env!("CARGO_PKG_VERSION")))
                .build()
                .map_err(|_| "cannot initialize HTTPS updater client")?,
            token: optional_token()?,
        })
    }

    fn get(&self, endpoint: &str, asset: bool) -> Result<Response> {
        let mut url = Url::parse(endpoint).map_err(|_| "invalid fixed GitHub API endpoint")?;
        if !api_url(&url) {
            return Err("refusing non-official release API endpoint".into());
        }
        for hop in 0..=MAX_REDIRECTS {
            if !trusted_url(&url) {
                return Err("GitHub redirected to an untrusted or non-HTTPS host".into());
            }
            let mut request = self
                .client
                .get(url.clone())
                .timeout(Duration::from_secs(if asset { 300 } else { 30 }))
                .header(
                    header::ACCEPT,
                    if asset {
                        "application/octet-stream"
                    } else {
                        "application/vnd.github+json"
                    },
                )
                .header("X-GitHub-Api-Version", "2022-11-28");
            // Do not send credentials on redirects, even to another API endpoint.
            if hop == 0 {
                if let Some(token) = &self.token {
                    request = request.header(header::AUTHORIZATION, token.clone());
                }
            }
            let response = request.send().map_err(
                |_| "GitHub HTTPS request failed (check connectivity, TLS, and proxy settings)",
            )?;
            if matches!(response.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
                if hop == MAX_REDIRECTS {
                    return Err("too many GitHub download redirects".into());
                }
                let location = response
                    .headers()
                    .get(header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or("GitHub redirect has no valid Location")?;
                url = url.join(location).map_err(|_| "invalid GitHub redirect")?;
                if !asset && !api_url(&url) {
                    return Err("release metadata redirected outside the fixed official API".into());
                }
                continue;
            }
            match response.status() {
                StatusCode::OK => return Ok(response),
                StatusCode::NOT_FOUND => return Err("GitHub release/asset not found (404). Hexadecimall/lma is private: set GH_TOKEN or GITHUB_TOKEN with repository Contents read access, and confirm a published stable release/tag and its assets exist. An empty release list is not an update.".into()),
                StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => return Err("GitHub denied release access; check GH_TOKEN/GITHUB_TOKEN repository Contents read permission and API rate limits".into()),
                _ => return Err(format!("GitHub returned HTTP {}; retry later or check release access", response.status().as_u16())),
            }
        }
        Err("too many GitHub redirects".into())
    }

    fn release(&self, version: Option<Version>) -> Result<Release> {
        let suffix = version.map_or_else(|| "latest".into(), |v| format!("tags/{}", v.tag()));
        let response = self.get(&format!("{API}/releases/{suffix}"), false)?;
        if response.content_length().is_some_and(|n| n > MAX_METADATA) {
            return Err("release metadata exceeds 8 MiB".into());
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_METADATA + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "cannot read release metadata")?;
        if bytes.len() as u64 > MAX_METADATA {
            return Err("release metadata exceeds 8 MiB".into());
        }
        serde_json::from_slice(&bytes)
            .map_err(
                |_| "GitHub returned invalid release metadata (expected one published release)",
            )
            .map_err(Into::into)
    }

    fn download(
        &self,
        asset: &Asset,
        writer: impl Write,
        maximum: u64,
        digest: Option<[u8; 32]>,
    ) -> Result<[u8; 32]> {
        asset.validate(maximum)?;
        let response = self.get(&format!("{API}/releases/assets/{}", asset.id), true)?;
        if response
            .content_length()
            .is_some_and(|size| size != asset.size)
        {
            return Err("asset Content-Length differs from release metadata".into());
        }
        copy_verified(
            response,
            writer,
            asset.size,
            maximum,
            digest,
            asset.sha256()?,
        )
    }
}

fn validate_header(header: &[u8], platform: &Platform) -> Result<()> {
    let valid = match platform.os.as_str() {
        "macos" => {
            // Thin 64-bit little-endian Mach-O executable, not an archive/dylib/fat wrapper.
            header.len() >= 32
                && header[..4] == [0xcf, 0xfa, 0xed, 0xfe]
                && u32::from_le_bytes(header[4..8].try_into().unwrap()) == 0x0100_000c
                && u32::from_le_bytes(header[12..16].try_into().unwrap()) == 2
        }
        "linux" | "freebsd" => {
            let machine = match platform.arch.as_str() {
                "x86_64" => 62,
                "aarch64" => 183,
                _ => 0,
            };
            header.len() >= 64
                && &header[..4] == b"\x7fELF"
                && header[4] == 2
                && header[5] == 1
                && header[6] == 1
                && (header[7] == 0 || header[7] == if platform.os == "freebsd" { 9 } else { 3 })
                && matches!(
                    u16::from_le_bytes(header[16..18].try_into().unwrap()),
                    2 | 3
                )
                && u16::from_le_bytes(header[18..20].try_into().unwrap()) == machine
        }
        "windows" => {
            if header.len() < 64 || &header[..2] != b"MZ" {
                false
            } else {
                let offset = u32::from_le_bytes(header[60..64].try_into().unwrap()) as usize;
                let machine = match platform.arch.as_str() {
                    "x86_64" => 0x8664,
                    "aarch64" => 0xaa64,
                    "x86" => 0x14c,
                    _ => 0,
                };
                offset >= 64
                    && offset
                        .checked_add(26)
                        .is_some_and(|end| end <= header.len())
                    && &header[offset..offset + 4] == b"PE\0\0"
                    && u16::from_le_bytes(header[offset + 4..offset + 6].try_into().unwrap())
                        == machine
                    && u16::from_le_bytes(header[offset + 22..offset + 24].try_into().unwrap())
                        & 0x2002
                        == 2
                    && u16::from_le_bytes(header[offset + 24..offset + 26].try_into().unwrap())
                        == if platform.arch == "x86" { 0x10b } else { 0x20b }
            }
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err("download is not a native executable for this OS/architecture; original executable was not replaced".into())
    }
}

fn validate_file(file: &mut File, platform: &Platform) -> Result<()> {
    file.seek(SeekFrom::Start(0))
        .map_err(|_| "cannot inspect candidate executable")?;
    let mut header = Vec::new();
    (&mut *file)
        .take(64 * 1024)
        .read_to_end(&mut header)
        .map_err(|_| "cannot read candidate header")?;
    validate_header(&header, platform)
}

fn guard_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
    {
        return Err("executable path must be absolute and normalized".into());
    }
    let parts: Vec<_> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_ascii_lowercase())
        .collect();
    let protected = [
        "/usr/bin",
        "/usr/sbin",
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib32",
        "/usr/libexec",
        "/usr/share",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc",
        "/boot",
        "/System",
        "/Library",
        "/opt/local",
        "/nix/store",
        "/snap",
        "/var/lib/flatpak",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix));
    let managed = parts.iter().any(|s| {
        s == "cellar"
            || s == "caskroom"
            || s.ends_with(".app")
            || s == "lma-desktop"
            || s == "lma-desktop.exe"
            || matches!(
                s.as_str(),
                "windows"
                    | "windowsapps"
                    | "program files"
                    | "program files (x86)"
                    | "chocolatey"
                    | "scoop"
            )
    });
    let source_build = parts
        .windows(2)
        .any(|p| p[0] == "target" && (p[1] == "debug" || p[1] == "release"))
        || parts
            .windows(3)
            .any(|p| p[0] == "target" && (p[2] == "debug" || p[2] == "release"));
    let manifest = path.parent().and_then(Path::parent).is_some_and(|prefix| {
        // The installer owns this prefix, including its uninstall manifest.
        match fs::symlink_metadata(prefix.join("share/lma/package-manifest.json")) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            _ => true,
        }
    });
    if protected || managed || source_build || manifest {
        return Err("this is a package-managed, desktop, or source-build executable; reinstall with its installer/package manager, or run an installed standalone CLI instead".into());
    }
    Ok(())
}

fn is_link(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

/// No writable-by-other-user directories, symlinks, or untrusted owners in the
/// publication/state path. Root-owned sticky temp ancestors are safe for tests.
fn trusted_directory(path: &Path, owner: Option<u32>) -> Result<()> {
    for directory in path.ancestors() {
        let metadata = fs::symlink_metadata(directory)
            .map_err(|_| "cannot inspect update parent directory")?;
        if !metadata.is_dir() || is_link(&metadata) {
            return Err(
                "update parent directories must be real trusted directories, not symlinks".into(),
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let sticky_root = metadata.uid() == 0 && metadata.mode() & 0o1000 != 0;
            if (metadata.mode() & 0o022 != 0 && !sticky_root)
                || owner.is_some_and(|uid| metadata.uid() != 0 && metadata.uid() != uid)
            {
                return Err("update directory is writable by another user or has an untrusted owner; reinstall manually".into());
            }
        }
        #[cfg(not(unix))]
        let _ = owner;
    }
    Ok(())
}

#[derive(Debug)]
struct Original {
    reported: PathBuf,
    path: PathBuf,
    metadata: Metadata,
    digest: [u8; 32],
}

fn same_metadata(left: &Metadata, right: &Metadata) -> bool {
    if left.len() != right.len() || left.modified().ok() != right.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        left.dev() == right.dev()
            && left.ino() == right.ino()
            && left.mode() == right.mode()
            && left.uid() == right.uid()
            && left.gid() == right.gid()
            && left.ctime() == right.ctime()
            && left.ctime_nsec() == right.ctime_nsec()
    }
    #[cfg(not(unix))]
    {
        left.permissions().readonly() == right.permissions().readonly()
    }
}

fn file_digest(path: &Path, metadata: &Metadata) -> Result<[u8; 32]> {
    let file = File::open(path).map_err(|_| "cannot read original executable")?;
    let before = file
        .metadata()
        .map_err(|_| "cannot inspect original executable")?;
    if !same_metadata(&before, metadata) {
        return Err("original executable changed; retry from a fresh launch".into());
    }
    let digest = copy_verified(
        &file,
        std::io::sink(),
        metadata.len(),
        MAX_BINARY,
        None,
        None,
    )?;
    let after = file
        .metadata()
        .map_err(|_| "cannot inspect original executable")?;
    if !same_metadata(metadata, &after) {
        return Err("original executable changed while reading; retry from a fresh launch".into());
    }
    Ok(digest)
}

impl Original {
    fn capture(reported: PathBuf) -> Result<Self> {
        let path =
            fs::canonicalize(&reported).map_err(|_| "cannot resolve installed CLI executable")?;
        guard_path(&path)?;
        let metadata =
            fs::symlink_metadata(&path).map_err(|_| "cannot inspect installed CLI executable")?;
        if !metadata.is_file() || is_link(&metadata) {
            return Err("installed CLI must resolve to a real regular file".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.mode() & 0o7022 != 0 {
                return Err("refusing a setuid/setgid/sticky or group/other-writable executable; reinstall with safe permissions".into());
            }
        }
        trusted_directory(path.parent().ok_or("executable has no parent")?, None)?;
        let digest = file_digest(&path, &metadata)?;
        let original = Self {
            reported,
            path,
            metadata,
            digest,
        };
        original.verify()?;
        Ok(original)
    }

    fn verify(&self) -> Result<()> {
        if fs::canonicalize(&self.reported).map_err(|_| "CLI path changed during update")?
            != self.path
        {
            return Err(
                "CLI symlink changed during update; original executable was not replaced".into(),
            );
        }
        let now = fs::symlink_metadata(&self.path)
            .map_err(|_| "original executable disappeared during update")?;
        if !now.is_file()
            || is_link(&now)
            || !same_metadata(&self.metadata, &now)
            || file_digest(&self.path, &now)? != self.digest
        {
            return Err("original executable changed during update; refusing to replace it".into());
        }
        Ok(())
    }
}

struct UpdateLock {
    path: PathBuf,
    _file: File,
}
impl Drop for UpdateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn create_state(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        return Err("update state path must be absolute".into());
    }
    if let Some(parent) = path.parent() {
        if fs::symlink_metadata(parent).is_err() {
            create_state(parent)?;
        }
        trusted_directory(parent, None)?;
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !is_link(&metadata) => {}
        Ok(_) => {
            return Err("update state directory must not be a symlink or non-directory".into());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder
                .create(path)
                .map_err(|_| "cannot create ~/.lma/updates; check directory permissions")?;
        }
        Err(_) => return Err("cannot inspect update state directory".into()),
    }
    trusted_directory(path, None)
}

fn lock_state(path: &Path) -> Result<(UpdateLock, Option<u32>)> {
    create_state(path)?;
    let lock_path = path.join("update.lock");
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&lock_path).map_err(|_| "cannot acquire ~/.lma/updates/update.lock; another update may be active. If a previous updater crashed, remove that lock only after confirming no update is running")?;
    let mut lock = UpdateLock {
        path: lock_path,
        _file: file,
    };
    let owner = {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Some(
                lock._file
                    .metadata()
                    .map_err(|_| "cannot inspect update lock")?
                    .uid(),
            )
        }
        #[cfg(not(unix))]
        {
            None
        }
    };
    trusted_directory(path, owner)?;
    writeln!(lock._file, "pid={}", std::process::id()).map_err(|_| "cannot write update lock")?;
    Ok((lock, owner))
}

fn backup_original(original: &Original, state: &Path) -> Result<PathBuf> {
    original.verify()?;
    let mut backup = tempfile::Builder::new()
        .prefix("lma-backup-")
        .suffix(if cfg!(windows) { ".exe" } else { "" })
        .tempfile_in(state)
        .map_err(|_| "cannot create executable backup")?;
    let source = File::open(&original.path).map_err(|_| "cannot read executable for backup")?;
    copy_verified(
        source,
        backup.as_file_mut(),
        original.metadata.len(),
        MAX_BINARY,
        Some(original.digest),
        None,
    )?;
    backup
        .as_file()
        .sync_all()
        .map_err(|_| "cannot sync executable backup")?;
    original.verify()?;
    let (_, path) = backup
        .keep()
        .map_err(|_| "cannot retain executable backup")?;
    #[cfg(unix)]
    File::open(state).and_then(|directory| directory.sync_all())
        .map_err(|_| "backup retained but cannot sync backup directory; original executable was not replaced")?;
    Ok(path)
}

#[cfg(unix)]
fn publish_unix(
    original: &Original,
    candidate: NamedTempFile,
    backup: &Path,
    owner: Option<u32>,
) -> Result<()> {
    publish_unix_with(
        original,
        candidate,
        backup,
        owner,
        |candidate, destination| {
            candidate
                .persist(destination)
                .map(|_| ())
                .map_err(|error| error.error)
        },
    )
}

#[cfg(unix)]
fn publish_unix_with(
    original: &Original,
    candidate: NamedTempFile,
    backup: &Path,
    owner: Option<u32>,
    replace: impl FnOnce(NamedTempFile, &Path) -> std::io::Result<()>,
) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    trusted_directory(
        original.path.parent().ok_or("executable has no parent")?,
        owner,
    )?;
    // Preserve ordinary read/execute permissions, but never special or write-for-others bits.
    let mode = (original.metadata.permissions().mode() & 0o755) | 0o500;
    candidate
        .as_file()
        .set_permissions(fs::Permissions::from_mode(mode))
        .map_err(|_| "cannot set safe executable permissions")?;
    candidate
        .as_file()
        .sync_all()
        .map_err(|_| "cannot sync candidate executable")?;
    original.verify()?;
    // Parents are trusted and the original (including any invocation symlink) was
    // rechecked immediately before this same-filesystem atomic rename.
    replace(candidate, &original.path).map_err(|_| {
        format!(
            "atomic replacement failed; original executable is unchanged; backup retained at {}",
            backup.display()
        )
    })?;
    if File::open(original.path.parent().unwrap())
        .and_then(|f| f.sync_all())
        .is_err()
    {
        return Err(format!(
            "executable was replaced, but syncing its directory failed; verify the installation before rebooting; backup: {}",
            backup.display()
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn powershell_literal(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "''"))
}

pub(crate) fn run(config: crate::config::UpdateConfig) -> Result<()> {
    let current = Version::parse(env!("CARGO_PKG_VERSION"))?;
    let requested = config.version.as_deref().map(Version::parse).transpose()?;
    if requested.is_some_and(|version| version < current) {
        return Err(format!("refusing to downgrade installed {}", current.tag()));
    }
    if requested == Some(current) {
        println!(
            "lma {} is already current; nothing installed or downloaded.",
            current.tag()
        );
        return Ok(());
    }
    let platform = Platform::compiled()?;
    platform.ensure_runtime_requirements()?;
    let github = GitHub::new()?;
    let release = github.release(requested)?;
    let version = release.version(requested)?;
    if version < current {
        return Err(format!(
            "release {} is older than installed {}; refusing to downgrade",
            version.tag(),
            current.tag()
        ));
    }
    if version == current {
        println!(
            "lma {} is already current; nothing installed or downloaded.",
            current.tag()
        );
        return Ok(());
    }
    let name = platform.asset_name(version);
    let (binary, checksums) = release.assets(&name)?;
    if config.check {
        println!(
            "Update available: {} -> {} ({name}). Metadata checked only; not installed. Run lma u{} to download and verify SHA256SUMS before installation.",
            current.tag(),
            version.tag(),
            requested.map_or_else(String::new, |v| format!(" -v {}", v.tag()))
        );
        return Ok(());
    }

    let original = Original::capture(
        std::env::current_exe().map_err(|_| "cannot locate current CLI executable")?,
    )?;
    let models = crate::models::discovery::default_models_dir()?;
    let state = models
        .parent()
        .ok_or("default models directory has no state parent")?
        .join("updates");
    let (_lock, owner) = lock_state(&state)?;
    trusted_directory(
        original.path.parent().ok_or("executable has no parent")?,
        owner,
    )?;
    let mut sums = Vec::new();
    github.download(checksums, &mut sums, MAX_CHECKSUMS, None)?;
    let expected = checksum_entry(&sums, &name)?;
    if binary.sha256()?.is_some_and(|digest| digest != expected) {
        return Err("GitHub asset digest disagrees with SHA256SUMS; refusing update".into());
    }
    #[cfg(unix)]
    let staging = original.path.parent().unwrap();
    #[cfg(not(unix))]
    let staging = state.as_path();
    let mut candidate = tempfile::Builder::new().prefix(".lma-update-")
        .suffix(if cfg!(windows) { ".exe" } else { "" }).tempfile_in(staging)
        .map_err(|_| "cannot create update temporary file; check installation directory permissions (do not escalate privileges)")?;
    github.download(binary, candidate.as_file_mut(), MAX_BINARY, Some(expected))?;
    validate_file(candidate.as_file_mut(), &platform)?;
    candidate
        .as_file()
        .sync_all()
        .map_err(|_| "cannot sync verified update")?;
    let backup = backup_original(&original, &state)?;
    #[cfg(unix)]
    {
        publish_unix(&original, candidate, &backup, owner)?;
        println!(
            "Installed lma {}. Backup: {}. Resident servers are unchanged; the new executable takes effect on the next launch. SHA256SUMS is from the release publisher, not an independent signature.",
            version.tag(),
            backup.display()
        );
    }
    #[cfg(windows)]
    {
        original.verify()?;
        let (_, staged) = candidate
            .keep()
            .map_err(|_| "cannot retain verified Windows update")?;
        println!(
            "Verified lma {} is staged at {}. NOT INSTALLED. Backup: {}. Exit this CLI; after lma processes using this executable have exited naturally, manually run in PowerShell:\nCopy-Item -LiteralPath {} -Destination {}\nDo not interrupt resident servers. The next launch after replacement uses the update. SHA256SUMS is publisher-provided, not an independent signature.",
            version.tag(),
            staged.display(),
            backup.display(),
            powershell_literal(&staged),
            powershell_literal(&original.path)
        );
    }
    #[cfg(not(any(unix, windows)))]
    return Err("self-update installation is unsupported on this platform".into());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn digest(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }
    fn hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn version() -> Version {
        Version(1, 2, 3)
    }
    fn asset(name: &str, id: u64) -> Asset {
        Asset {
            id,
            name: name.into(),
            state: "uploaded".into(),
            size: 3,
            digest: None,
        }
    }
    fn release() -> Release {
        Release {
            tag_name: "v1.2.3".into(),
            draft: false,
            prerelease: false,
            assets: vec![asset("binary", 1), asset("SHA256SUMS", 2)],
        }
    }
    fn linux() -> Platform {
        Platform::select("linux", "x86_64", "gnu", false, false).unwrap()
    }
    fn elf() -> Vec<u8> {
        let mut bytes = vec![0; 64];
        bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        bytes[16] = 2;
        bytes[18] = 62;
        bytes
    }

    #[test]
    fn release_runtime_minimums_fail_closed_for_older_or_unknown_versions() {
        assert!(version_at_least("2.39", 2, 39));
        assert!(version_at_least("2.40.1", 2, 39));
        assert!(!version_at_least("2.35", 2, 39));
        assert!(!version_at_least("2.38.99", 2, 39));
        assert!(version_at_least("24.0.0", 24, 0));
        assert!(version_at_least("25.1.0", 24, 0));
        assert!(!version_at_least("23.6.0", 24, 0));
        for value in ["", "unknown", "2", "2.x", "-1.40"] {
            assert!(!version_at_least(value, 2, 39));
        }
    }

    #[test]
    fn strict_versions_and_numeric_ordering() {
        for valid in ["0.0.0", "1.2.3", "v1.2.3", "18446744073709551615.0.0"] {
            assert!(Version::parse(valid).is_ok(), "{valid}");
        }
        for invalid in [
            "",
            "1",
            "1.2",
            "1.2.3.4",
            "V1.2.3",
            "vv1.2.3",
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "1.2.3-rc1",
            "1.2.3+build",
            " 1.2.3",
            "1.2.3\n",
            "+1.2.3",
            "1.2.-3",
            "１.2.3",
            "18446744073709551616.0.0",
        ] {
            assert!(Version::parse(invalid).is_err(), "{invalid}");
        }
        assert!(Version::parse("0.10.0").unwrap() > Version::parse("0.9.99").unwrap());
        assert_eq!(Version::parse("1.2.3").unwrap().tag(), "v1.2.3");
    }

    #[test]
    fn stable_release_tag_must_match() {
        let mut r = release();
        assert_eq!(r.version(Some(version())).unwrap(), version());
        assert!(r.version(Some(Version(2, 0, 0))).is_err());
        r.draft = true;
        assert!(r.version(None).is_err());
        r.draft = false;
        r.prerelease = true;
        assert!(r.version(None).is_err());
        r.prerelease = false;
        r.tag_name = "1.2.3".into();
        assert!(r.version(None).is_err());
        r.tag_name = "v1.2.3-rc1".into();
        assert!(r.version(None).is_err());
    }

    #[test]
    fn exact_platform_and_compiled_backend_preservation() {
        let mac = Platform::select("macos", "aarch64", "", false, false).unwrap();
        assert_eq!(
            mac.asset_name(version()),
            "lma-v1.2.3-aarch64-apple-darwin-metal"
        );
        assert_eq!(
            linux().asset_name(version()),
            "lma-v1.2.3-x86_64-unknown-linux-gnu-cpu"
        );
        let cuda = Platform::select("linux", "aarch64", "musl", false, true).unwrap();
        assert_eq!(
            cuda.asset_name(version()),
            "lma-v1.2.3-aarch64-unknown-linux-musl-cuda"
        );
        let windows = Platform::select("windows", "x86_64", "msvc", false, false).unwrap();
        assert_eq!(
            windows.asset_name(version()),
            "lma-v1.2.3-x86_64-pc-windows-msvc-cpu.exe"
        );
        assert!(
            Platform::select("windows", "x86_64", "gnu", false, true)
                .unwrap()
                .asset_name(version())
                .ends_with("gnu-cuda.exe")
        );
        assert_eq!(
            Platform::select("freebsd", "x86_64", "", false, false)
                .unwrap()
                .target,
            "x86_64-unknown-freebsd"
        );
        for args in [
            ("macos", "aarch64", "", false, true),
            ("linux", "x86_64", "gnu", true, false),
            ("linux", "x86_64", "gnu", true, true),
            ("linux", "riscv64", "gnu", false, false),
            ("windows", "x86_64", "", false, false),
            ("freebsd", "x86_64", "", false, true),
        ] {
            assert!(Platform::select(args.0, args.1, args.2, args.3, args.4).is_err());
        }
        let r = release();
        assert!(r.assets(&cuda.asset_name(version())).is_err());
    }

    #[test]
    fn asset_selection_requires_unique_uploaded_bounded_assets() {
        let mut r = release();
        assert!(r.assets("binary").is_ok());
        assert!(r.assets("binary.zip").is_err());
        r.assets.push(asset("binary", 3));
        assert!(r.assets("binary").is_err());
        r.assets.pop();
        r.assets[1].id = 1;
        assert!(r.assets("binary").is_err());
        r.assets[1].id = 2;
        r.assets[0].id = 0;
        assert!(r.assets("binary").is_err());
        r.assets[0].id = 1;
        for size in [0, MAX_BINARY + 1] {
            r.assets[0].size = size;
            assert!(r.assets("binary").is_err());
        }
        r.assets[0].size = MAX_BINARY;
        assert!(r.assets("binary").is_ok());
        r.assets[1].size = MAX_CHECKSUMS + 1;
        assert!(r.assets("binary").is_err());
        r.assets[1].size = 3;
        r.assets[0].state = "new".into();
        assert!(r.assets("binary").is_err());
        r.assets.pop();
        assert!(r.assets("binary").is_err());
    }

    #[test]
    fn optional_api_digests_are_strict_and_verified() {
        let mut a = asset("binary", 1);
        assert_eq!(a.sha256().unwrap(), None);
        a.digest = Some(format!("sha256:{}", hex(b"abc")));
        assert_eq!(a.sha256().unwrap(), Some(digest(b"abc")));
        for invalid in ["sha512:abc", "sha256:abc", "", "SHA256:abc"] {
            a.digest = Some(invalid.into());
            assert!(a.sha256().is_err());
        }
        assert!(parse_hex(&"z".repeat(64)).is_err());
        assert_eq!(
            parse_hex(&hex(b"abc").to_uppercase()).unwrap(),
            digest(b"abc")
        );
        assert!(
            copy_verified(
                Cursor::new(b"abc"),
                Vec::new(),
                3,
                3,
                Some(digest(b"abc")),
                Some(digest(b"xyz"))
            )
            .is_err()
        );
    }

    #[test]
    fn checksums_require_one_safe_unambiguous_record() {
        let good = format!("{}  other\n{} *binary\n", hex(b"xyz"), hex(b"abc"));
        assert_eq!(
            checksum_entry(good.as_bytes(), "binary").unwrap(),
            digest(b"abc")
        );
        for name in [
            "../binary",
            "/binary",
            "a/binary",
            "a\\binary",
            ".",
            "..",
            "C:binary",
            "bi nary",
            "binary\t",
            "bínary",
        ] {
            let text = format!("{}  {name}\n", hex(b"abc"));
            assert!(checksum_entry(text.as_bytes(), "binary").is_err(), "{name}");
        }
        for text in [
            format!("{good}{good}"),
            format!("{} binary\n", hex(b"abc")),
            format!("{}  other\n", hex(b"abc")),
            String::new(),
            "# comment\n".into(),
            format!("\\{}  binary\n", hex(b"abc")),
            format!(
                "{}  binary\n{}  other\n{}  other\n",
                hex(b"abc"),
                hex(b"abc"),
                hex(b"xyz")
            ),
        ] {
            assert!(checksum_entry(text.as_bytes(), "binary").is_err());
        }
        assert!(checksum_entry(&[0xff], "binary").is_err());
    }

    struct FailingReader;
    impl Read for FailingReader {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected network failure"))
        }
    }
    struct FailingWriter;
    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn streaming_verification_bounds_reads_and_preserves_outputs_on_errors() {
        let mut output = Vec::new();
        assert_eq!(
            copy_verified(
                Cursor::new(b"abc"),
                &mut output,
                3,
                3,
                Some(digest(b"abc")),
                None
            )
            .unwrap(),
            digest(b"abc")
        );
        assert_eq!(output, b"abc");
        assert!(copy_verified(Cursor::new(b"ab"), Vec::new(), 3, 3, None, None).is_err());
        let mut input = Cursor::new(b"abcdef");
        assert!(copy_verified(&mut input, Vec::new(), 3, 3, None, None).is_err());
        assert_eq!(input.position(), 4);
        assert!(copy_verified(Cursor::new(b"abc"), Vec::new(), 4, 3, None, None).is_err());
        assert!(copy_verified(Cursor::new(b""), Vec::new(), 0, 3, None, None).is_err());
        assert!(copy_verified(FailingReader, Vec::new(), 3, 3, None, None).is_err());
        assert!(copy_verified(Cursor::new(b"abc"), FailingWriter, 3, 3, None, None).is_err());
        assert!(
            copy_verified(
                Cursor::new(b"abc"),
                Vec::new(),
                3,
                3,
                Some(digest(b"xyz")),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn redirects_only_allow_fixed_https_hosts_without_userinfo() {
        for value in [
            "https://api.github.com/repos/Hexadecimall/lma/releases/latest",
            "https://github.com/Hexadecimall/lma/releases/download/v1.2.3/binary",
            "https://objects.githubusercontent.com/path?signature=opaque",
            "https://release-assets.githubusercontent.com/path",
        ] {
            assert!(trusted_url(&Url::parse(value).unwrap()));
        }
        for value in [
            "http://api.github.com/path",
            "https://evil.example/path",
            "https://api.github.com.evil.example/path",
            "https://api.github.com:444/path",
            "https://user:secret@api.github.com/path",
            "https://api.github.com/path#fragment",
            "https://127.0.0.1/path",
        ] {
            assert!(!trusted_url(&Url::parse(value).unwrap()));
        }
        assert!(api_url(
            &Url::parse(&format!("{API}/releases/latest")).unwrap()
        ));
        for value in [
            "https://api.github.com/repos/other/repo/releases/latest",
            "https://github.com/repos/Hexadecimall/lma/releases/latest",
            "https://api.github.com/repos/Hexadecimall/lma/releases/latest?token=secret",
        ] {
            assert!(!api_url(&Url::parse(value).unwrap()));
        }
    }

    #[test]
    fn native_headers_reject_text_wrong_os_arch_and_libraries() {
        let bytes = elf();
        assert!(validate_header(&bytes, &linux()).is_ok());
        for invalid in [
            b"#!/bin/sh\necho bad".to_vec(),
            b"<html>error</html>".to_vec(),
            vec![],
            bytes[..20].to_vec(),
        ] {
            assert!(validate_header(&invalid, &linux()).is_err());
        }
        for (index, value) in [(4, 1), (5, 2), (7, 9), (16, 1), (18, 183)] {
            let mut bad = bytes.clone();
            bad[index] = value;
            assert!(validate_header(&bad, &linux()).is_err());
        }
        let mac = Platform::select("macos", "aarch64", "", false, false).unwrap();
        let mut macho = vec![0; 32];
        macho[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
        macho[4..8].copy_from_slice(&0x0100_000cu32.to_le_bytes());
        macho[12] = 2;
        assert!(validate_header(&macho, &mac).is_ok());
        assert!(validate_header(&macho, &linux()).is_err());
        macho[4] = 7;
        assert!(validate_header(&macho, &mac).is_err());
        let win = Platform::select("windows", "x86_64", "msvc", false, false).unwrap();
        let mut pe = vec![0; 128];
        pe[..2].copy_from_slice(b"MZ");
        pe[60] = 64;
        pe[64..68].copy_from_slice(b"PE\0\0");
        pe[68..70].copy_from_slice(&0x8664u16.to_le_bytes());
        pe[86] = 2;
        pe[88..90].copy_from_slice(&0x20bu16.to_le_bytes());
        assert!(validate_header(&pe, &win).is_ok());
        pe[87] = 0x20;
        assert!(validate_header(&pe, &win).is_err());
        pe[60..64].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(validate_header(&pe, &win).is_err());
    }

    #[test]
    fn path_guards_protect_managed_desktop_and_build_paths() {
        for path in [
            "/usr/bin/lma",
            "/usr/sbin/lma",
            "/bin/lma",
            "/nix/store/pkg/bin/lma",
            "/opt/homebrew/Cellar/lma/1/bin/lma",
            "/usr/local/Caskroom/lma/bin/lma",
            "/Applications/lma.app/Contents/MacOS/lma",
            "/project/target/debug/lma",
            "/project/target/release/lma",
            "/project/target/x86_64-unknown-linux-gnu/release/lma",
            "/home/user/.local/bin/lma-desktop",
            "/home/user/.local/bin/lma-desktop.exe",
        ] {
            assert!(guard_path(Path::new(path)).is_err(), "{path}");
        }
        assert!(guard_path(Path::new("relative/lma")).is_err());
        let temp = tempfile::tempdir().unwrap();
        let prefix = fs::canonicalize(temp.path()).unwrap();
        fs::create_dir(prefix.join("bin")).unwrap();
        let executable = prefix.join("bin/lma");
        assert!(guard_path(&executable).is_ok());
        fs::create_dir_all(prefix.join("share/lma")).unwrap();
        assert!(
            guard_path(&executable).is_ok(),
            "legacy user resources do not imply a managed package"
        );
        fs::write(prefix.join("share/lma/package-manifest.json"), b"{}").unwrap();
        assert!(guard_path(&executable).is_err());
    }

    fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temp.path()).unwrap();
        let path = root.join("lma");
        fs::write(&path, b"original executable").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let state = root.join("updates");
        (temp, path, state)
    }

    #[test]
    fn state_lock_is_exclusive_and_released() {
        let (_temp, _path, state) = fixture();
        let (lock, _) = lock_state(&state).unwrap();
        assert!(lock_state(&state).is_err());
        drop(lock);
        assert!(!state.join("update.lock").exists());
        assert!(lock_state(&state).is_ok());
    }

    #[test]
    fn backup_is_retained_and_original_mutation_is_detected() {
        let (_temp, path, state) = fixture();
        let original = Original::capture(path.clone()).unwrap();
        let (_lock, _) = lock_state(&state).unwrap();
        let backup = backup_original(&original, &state).unwrap();
        assert_eq!(fs::read(backup).unwrap(), b"original executable");
        fs::write(&path, b"changed executable!").unwrap();
        assert!(original.verify().is_err());
        assert!(backup_original(&original, &state).is_err());
        assert_eq!(fs::read(path).unwrap(), b"changed executable!");
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_resolved_but_retargeting_is_rejected() {
        use std::os::unix::fs::symlink;
        let (_temp, path, state) = fixture();
        let link = path.parent().unwrap().join("cli-link");
        symlink(&path, &link).unwrap();
        let original = Original::capture(link.clone()).unwrap();
        assert_eq!(original.path, path);
        let other = path.parent().unwrap().join("other");
        fs::write(&other, b"other").unwrap();
        fs::remove_file(&link).unwrap();
        symlink(other, &link).unwrap();
        assert!(original.verify().is_err());
        assert!(!state.exists());
    }

    #[cfg(unix)]
    #[test]
    fn unsafe_permissions_and_symlink_state_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let (_temp, path, state) = fixture();
        for mode in [0o4755, 0o2755, 0o777] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
            assert!(Original::capture(path.clone()).is_err());
        }
        let real = path.parent().unwrap().join("real-state");
        fs::create_dir(&real).unwrap();
        symlink(&real, &state).unwrap();
        assert!(lock_state(&state).is_err());
        fs::set_permissions(&real, fs::Permissions::from_mode(0o777)).unwrap();
        assert!(trusted_directory(&real, None).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_publish_retains_backup_and_safe_mode() {
        use std::os::unix::fs::PermissionsExt;
        let (_temp, path, state) = fixture();
        let original = Original::capture(path.clone()).unwrap();
        let (_lock, owner) = lock_state(&state).unwrap();
        let backup = backup_original(&original, &state).unwrap();
        let mut candidate = NamedTempFile::new_in(path.parent().unwrap()).unwrap();
        candidate.write_all(b"verified replacement").unwrap();
        publish_unix(&original, candidate, &backup, owner).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"verified replacement");
        assert_eq!(fs::read(&backup).unwrap(), b"original executable");
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o7777,
            0o755
        );
    }

    #[cfg(unix)]
    #[test]
    fn injected_rename_failure_preserves_original_and_retained_backup() {
        let (_temp, path, state) = fixture();
        let original = Original::capture(path.clone()).unwrap();
        let (_lock, owner) = lock_state(&state).unwrap();
        let backup = backup_original(&original, &state).unwrap();
        let mut candidate = NamedTempFile::new_in(path.parent().unwrap()).unwrap();
        candidate.write_all(b"verified replacement").unwrap();
        let temporary = candidate.path().to_owned();
        let error = publish_unix_with(&original, candidate, &backup, owner, |_, _| {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected rename failure",
            ))
        })
        .unwrap_err();
        assert!(error.contains("atomic replacement failed"));
        original.verify().unwrap();
        assert_eq!(fs::read(path).unwrap(), b"original executable");
        assert_eq!(fs::read(backup).unwrap(), b"original executable");
        assert!(!temporary.exists());
    }

    #[test]
    fn malformed_release_metadata_is_not_an_update() {
        for json in [
            r#"[]"#,
            r#"{}"#,
            r#"{"tag_name":"v1.2.3","draft":false,"prerelease":false,"assets":[{"id":-1,"name":"binary","size":3,"state":"uploaded"}]}"#,
            r#"{"tag_name":"v1.2.3","tag_name":"v9.0.0","draft":false,"prerelease":false,"assets":[]}"#,
        ] {
            assert!(serde_json::from_str::<Release>(json).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn failed_atomic_publish_preserves_changed_original_and_backup() {
        let (_temp, path, state) = fixture();
        let original = Original::capture(path.clone()).unwrap();
        let (_lock, owner) = lock_state(&state).unwrap();
        let backup = backup_original(&original, &state).unwrap();
        let mut candidate = NamedTempFile::new_in(path.parent().unwrap()).unwrap();
        candidate.write_all(b"replacement").unwrap();
        let temporary = candidate.path().to_owned();
        fs::write(&path, b"concurrent installation").unwrap();
        assert!(publish_unix(&original, candidate, &backup, owner).is_err());
        assert_eq!(fs::read(path).unwrap(), b"concurrent installation");
        assert_eq!(fs::read(backup).unwrap(), b"original executable");
        assert!(!temporary.exists());
    }

    #[cfg(unix)]
    #[test]
    fn replaced_original_with_symlink_is_never_overwritten() {
        use std::os::unix::fs::symlink;
        let (_temp, path, state) = fixture();
        let original = Original::capture(path.clone()).unwrap();
        let (_lock, owner) = lock_state(&state).unwrap();
        let backup = backup_original(&original, &state).unwrap();
        let victim = path.parent().unwrap().join("victim");
        fs::write(&victim, b"do not touch").unwrap();
        fs::remove_file(&path).unwrap();
        symlink(&victim, &path).unwrap();
        let candidate = NamedTempFile::new_in(path.parent().unwrap()).unwrap();
        assert!(publish_unix(&original, candidate, &backup, owner).is_err());
        assert!(fs::symlink_metadata(path).unwrap().file_type().is_symlink());
        assert_eq!(fs::read(victim).unwrap(), b"do not touch");
    }

    #[test]
    fn failed_stream_does_not_touch_original_or_leave_candidate() {
        let (_temp, path, _state) = fixture();
        let original = Original::capture(path.clone()).unwrap();
        let temporary;
        {
            let mut candidate = NamedTempFile::new_in(path.parent().unwrap()).unwrap();
            temporary = candidate.path().to_owned();
            assert!(
                copy_verified(
                    FailingReader,
                    candidate.as_file_mut(),
                    3,
                    MAX_BINARY,
                    None,
                    None
                )
                .is_err()
            );
        }
        original.verify().unwrap();
        assert_eq!(fs::read(path).unwrap(), b"original executable");
        assert!(!temporary.exists());
    }

    #[cfg(windows)]
    #[test]
    fn windows_instructions_quote_literal_paths() {
        assert_eq!(
            powershell_literal(Path::new("C:\\Users\\O'Brien\\lma.exe")),
            "'C:\\Users\\O''Brien\\lma.exe'"
        );
    }
}
