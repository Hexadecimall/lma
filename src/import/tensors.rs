//! Strict, streaming codec for Ollama's per-tensor MLX safetensors blobs.
//!
//! Shapes remain in safetensors (row-major) order; the GGUF writer reverses them.
//! NVFP4 `global_scale` belongs on the parent's projection `.scale` tensor. When
//! `embedding_q8` is true it is instead folded into Q8_0 values, exactly once.
//! No model-name mapping, GGUF writing, store writes, or backend initialization
//! is performed here. Only the requested name and its NVFP4 companions are legal.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{File, Metadata};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::SystemTime;

use llama_cpp_sys_2 as ggml;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

const HEADER_LIMIT: u64 = 1024 * 1024;
const IO_CHUNK: usize = 1024 * 1024;
// A row can itself be untrusted/huge. Process aligned portions of a row, with
// at most 1 MiB of floats rather than allocating a full embedding or huge row.
const CHUNK_ELEMENTS: usize = IO_CHUNK / size_of::<f32>();
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Actual, unpacked dimensions, in source order. Scalars retain shape `[]`.
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    /// Native output length (use `output_info` for the embedding Q8_0 path).
    pub byte_len: u64,
    pub global_scale: Option<f32>,
    name: String,
    layout: Layout,
    source: Source,
    q8_byte_len: u64,
}

#[derive(Clone, Debug, PartialEq)]
enum Layout {
    Plain {
        data_start: u64,
        weight: Descriptor,
    },
    Nvfp4 {
        data_start: u64,
        weight: Descriptor,
        scales: Descriptor,
        global: Descriptor,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Source {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    ctime: (i64, i64),
    #[cfg(windows)]
    created: u64,
    #[cfg(windows)]
    attributes: u32,
    #[cfg(windows)]
    file_id: Option<(u32, u64)>,
}

impl Source {
    fn of(meta: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(windows)]
        use std::os::windows::fs::MetadataExt;
        Self {
            len: meta.len(),
            modified: meta.modified().ok(),
            #[cfg(unix)]
            dev: meta.dev(),
            #[cfg(unix)]
            ino: meta.ino(),
            #[cfg(unix)]
            ctime: (meta.ctime(), meta.ctime_nsec()),
            #[cfg(windows)]
            created: meta.creation_time(),
            #[cfg(windows)]
            attributes: meta.file_attributes(),
            #[cfg(windows)]
            file_id: None,
        }
    }

    pub(super) fn of_file(file: &File) -> Result<Self> {
        let source = Self::of(&file.metadata().map_err(io_error)?);
        #[cfg(windows)]
        let source = Self {
            file_id: Some(windows_file_id(file)?),
            ..source
        };
        Ok(source)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
enum Dtype {
    BF16,
    F32,
    F16,
    U32,
    U8,
}

impl Dtype {
    fn width(self) -> u64 {
        match self {
            Self::BF16 | Self::F16 => 2,
            Self::F32 | Self::U32 => 4,
            Self::U8 => 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    dtype: Dtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantMetadata {
    quant_type: String,
}

struct Header {
    metadata: Option<QuantMetadata>,
    tensors: BTreeMap<String, Descriptor>,
}

// serde_json::Value/maps silently overwrite duplicate keys. A duplicate tensor,
// descriptor field, or metadata field must never change what we validate.
impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a per-tensor safetensors header")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Header, A::Error> {
                let mut header = Header {
                    metadata: None,
                    tensors: BTreeMap::new(),
                };
                let mut seen = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom(format!("duplicate header key {key:?}")));
                    }
                    if key == "__metadata__" {
                        header.metadata = Some(map.next_value()?);
                    } else {
                        header.tensors.insert(key, map.next_value()?);
                    }
                }
                Ok(header)
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

/// Inspect and fully validate a single installed blob, without producing output.
/// Paths must not contain symlinks (including ancestor directories) or `..`.
/// Only regular files are accepted. Windows additionally rejects reparse points
/// and requires trusted, non-racing parent directories; see `open_regular`.
pub fn inspect(path: &Path, name: &str) -> Result<Tensor> {
    let mut file = open_regular(path)?;
    inspect_file(&mut file, name).map_err(|e| format!("{}: {e}", path.display()))
}

fn inspect_file(file: &mut File, name: &str) -> Result<Tensor> {
    if name.is_empty()
        || name == "__metadata__"
        || name.ends_with(".global_scale")
        || name.ends_with(".scale")
    {
        return Err("expected a weight tensor name, not a scale companion".into());
    }
    let source = Source::of_file(file)?;
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut size = [0; 8];
    file.read_exact(&mut size).map_err(io_error)?;
    let header_len = u64::from_le_bytes(size);
    if header_len == 0 || header_len > HEADER_LIMIT {
        return Err("safetensors header length must be in 1..=1 MiB".into());
    }
    let data_start = add(8, header_len)?;
    let payload_len = source
        .len
        .checked_sub(data_start)
        .ok_or("truncated safetensors header")?;
    let mut bytes = vec![0; header_len as usize];
    file.read_exact(&mut bytes).map_err(io_error)?;
    if bytes.first() != Some(&b'{') {
        return Err("safetensors header must start with a JSON object".into());
    }
    let mut header: Header =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid safetensors header: {e}"))?;
    let weight = header
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor {name:?}"))?
        .clone();
    // Validate every range before seeking into any payload. Sorted ranges must
    // cover the entire data section exactly: no aliases, holes, or trailers.
    let mut ranges = Vec::new();
    for (key, desc) in &header.tensors {
        let count = elements(&desc.shape)?;
        let len = mul(count, desc.dtype.width())?;
        let [start, end] = desc.data_offsets;
        if end.checked_sub(start) != Some(len) || end > payload_len {
            return Err(format!("invalid data_offsets/byte length for {key:?}"));
        }
        ranges.push((start, end));
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (start, next) in ranges {
        if start != end {
            return Err("tensor ranges overlap or contain gaps".into());
        }
        end = next;
    }
    if end != payload_len {
        return Err("unclaimed/truncated safetensors payload".into());
    }
    header.tensors.remove(name);
    if weight.dtype != Dtype::U32 {
        if header.metadata.is_some() || !header.tensors.is_empty() {
            return Err("plain tensor has unexpected metadata or companion tensors".into());
        }
        let ggml_type = match weight.dtype {
            Dtype::BF16 => ggml::GGML_TYPE_BF16,
            Dtype::F16 => ggml::GGML_TYPE_F16,
            Dtype::F32 => ggml::GGML_TYPE_F32,
            _ => return Err("plain tensors must be BF16, F16, or F32".into()),
        };
        return Ok(Tensor {
            shape: weight.shape.clone(),
            ggml_type,
            byte_len: weight.data_offsets[1] - weight.data_offsets[0],
            global_scale: None,
            name: name.into(),
            layout: Layout::Plain { data_start, weight },
            source,
            q8_byte_len: 0,
        });
    }
    if header.metadata.as_ref().map(|m| m.quant_type.as_str()) != Some("nvfp4") {
        return Err("U32 weights require quant_type=nvfp4".into());
    }
    let scales = header
        .tensors
        .remove(&format!("{name}.scale"))
        .ok_or("missing NVFP4 scale tensor")?;
    let global = header
        .tensors
        .remove(&format!("{name}.global_scale"))
        .ok_or("missing NVFP4 global_scale tensor")?;
    if !header.tensors.is_empty() {
        return Err("unexpected tensor names in NVFP4 blob".into());
    }
    let [rows, packed_cols] = weight.shape.as_slice() else {
        return Err("NVFP4 weight must have rank 2".into());
    };
    let cols = mul(*packed_cols, 8)?;
    if cols > i64::MAX as u64 || cols % 64 != 0 {
        return Err("NVFP4 unpacked columns must fit int64 and be divisible by 64".into());
    }
    let count = mul(*rows, cols)?;
    if count > i64::MAX as u64 {
        return Err("NVFP4 element count exceeds int64".into());
    }
    if scales.dtype != Dtype::U8 || scales.shape != [*rows, cols / 16] {
        return Err("NVFP4 scales must be U8 [rows, columns/16]".into());
    }
    if global.dtype != Dtype::F32 || !global.shape.is_empty() {
        return Err("NVFP4 global_scale must be an F32 scalar (shape [])".into());
    }
    seek_tensor(file, data_start, &global)?;
    let mut value = [0; 4];
    file.read_exact(&mut value).map_err(io_error)?;
    let global_scale = f32::from_le_bytes(value);
    if !global_scale.is_finite() {
        return Err("nonfinite NVFP4 global_scale".into());
    }
    seek_tensor(file, data_start, &scales)?;
    validate_scales(file, scales.data_offsets[1] - scales.data_offsets[0])?;
    let byte_len = mul(count / 64, 36)?;
    let q8_byte_len = mul(count / 32, 34)?;
    let shape = vec![*rows, cols];
    Ok(Tensor {
        shape,
        ggml_type: ggml::GGML_TYPE_NVFP4,
        byte_len,
        global_scale: Some(global_scale),
        name: name.into(),
        layout: Layout::Nvfp4 {
            data_start,
            weight,
            scales,
            global,
        },
        source,
        q8_byte_len,
    })
}

impl Tensor {
    /// Plain tensors are unchanged in either mode. For NVFP4 embeddings this
    /// returns Q8_0 size/type; do NOT also emit/apply `global_scale` in that mode.
    pub fn output_info(&self, embedding_q8: bool) -> (u32, u64) {
        if embedding_q8 && matches!(self.layout, Layout::Nvfp4 { .. }) {
            (ggml::GGML_TYPE_Q8_0, self.q8_byte_len)
        } else {
            (self.ggml_type, self.byte_len)
        }
    }

    /// Reopen and revalidate before touching `output`. Input-validation errors
    /// emit no bytes. I/O failures (or concurrent writes after validation) can
    /// still interrupt streaming, so the parent must publish its GGUF atomically.
    pub fn write(&self, path: &Path, output: &mut dyn Write, embedding_q8: bool) -> Result<()> {
        let mut weights_file = open_regular(path)?;
        let current = inspect_file(&mut weights_file, &self.name)?;
        if current != *self {
            return Err("tensor source or descriptor changed since inspect".into());
        }
        match &self.layout {
            Layout::Plain { data_start, weight } => {
                seek_tensor(&mut weights_file, *data_start, weight)?;
                let mut output = BufWriter::with_capacity(IO_CHUNK, output);
                copy_exact(&mut weights_file, &mut output, self.byte_len)?;
                output.flush().map_err(io_error)
            }
            Layout::Nvfp4 {
                data_start,
                weight,
                scales,
                ..
            } => {
                // A real second open, not try_clone (which shares a seek cursor).
                let mut scales_file = open_regular(path)?;
                if Source::of_file(&scales_file)? != self.source {
                    return Err("tensor source changed while opening scales".into());
                }
                if embedding_q8 {
                    // Validate actual dequantized values/Q8 representability in
                    // a bounded preflight pass, before emitting even one byte.
                    self.for_chunks(
                        &mut weights_file,
                        &mut scales_file,
                        *data_start,
                        weight,
                        scales,
                        |w, s| {
                            let values = dequantize(w, s, self.global_scale.unwrap());
                            validate_q8(&values)
                        },
                    )?;
                }
                let mut output = BufWriter::with_capacity(IO_CHUNK, output);
                self.for_chunks(
                    &mut weights_file,
                    &mut scales_file,
                    *data_start,
                    weight,
                    scales,
                    |w, s| {
                        if embedding_q8 {
                            let values = dequantize(w, s, self.global_scale.unwrap());
                            write_q8(&values, &mut output)
                        } else {
                            let packed = repack(w, s);
                            output.write_all(&packed).map_err(io_error)
                        }
                    },
                )?;
                output.flush().map_err(io_error)
            }
        }
    }

    fn for_chunks(
        &self,
        weights_file: &mut File,
        scales_file: &mut File,
        data_start: u64,
        weight: &Descriptor,
        scales: &Descriptor,
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        seek_tensor(weights_file, data_start, weight)?;
        seek_tensor(scales_file, data_start, scales)?;
        let cols = self.shape[1];
        let capacity = cols.min(CHUNK_ELEMENTS as u64) as usize;
        let mut w = vec![0; capacity / 2];
        let mut s = vec![0; capacity / 16];
        for _ in 0..self.shape[0] {
            let mut remaining = cols;
            while remaining != 0 {
                let n = remaining.min(CHUNK_ELEMENTS as u64) as usize;
                weights_file.read_exact(&mut w[..n / 2]).map_err(io_error)?;
                scales_file.read_exact(&mut s[..n / 16]).map_err(io_error)?;
                visit(&w[..n / 2], &s[..n / 16])?;
                remaining -= n as u64;
            }
        }
        Ok(())
    }
}

fn repack(weights: &[u8], scales: &[u8]) -> Vec<u8> {
    let mut output = vec![0; weights.len() / 32 * 36];
    for ((w, s), out) in weights
        .as_chunks::<32>()
        .0
        .iter()
        .zip(scales.as_chunks::<4>().0)
        .zip(output.as_chunks_mut::<36>().0)
    {
        out[..4].copy_from_slice(s);
        for sub in 0..4 {
            let source = &w[sub * 8..sub * 8 + 8];
            for j in 0..8 {
                let nibble = |i: usize| (source[i / 2] >> (4 * (i % 2))) & 15;
                out[4 + sub * 8 + j] = nibble(j) | (nibble(j + 8) << 4);
            }
        }
    }
    output
}

fn e4m3(scale: u8) -> f32 {
    let exp = (scale >> 3) & 15;
    let mantissa = scale & 7;
    if exp == 0 {
        f32::from(mantissa) / 512.0
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exp) - 7)
    }
}

fn dequantize(weights: &[u8], scales: &[u8], global: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(weights.len() * 2);
    for (sub, &scale) in weights.as_chunks::<8>().0.iter().zip(scales) {
        let scale = e4m3(scale);
        for &byte in sub {
            values.push((E2M1[(byte & 15) as usize] * scale) * global);
            values.push((E2M1[(byte >> 4) as usize] * scale) * global);
        }
    }
    values
}

fn validate_q8(values: &[f32]) -> Result<()> {
    for block in values.as_chunks::<32>().0 {
        let mut amax = 0.0_f32;
        for &value in block {
            if !value.is_finite() {
                return Err("nonfinite dequantized embedding value".into());
            }
            amax = amax.max(value.abs());
        }
        let d = amax / 127.0;
        // Q8_0 stores d as fp16. Also avoid overflowing the reciprocal used by
        // ggml's reference quantizer for tiny/subnormal nonzero inputs.
        if d > 65504.0 || (amax != 0.0 && (d == 0.0 || !(1.0 / d).is_finite())) {
            return Err("embedding values cannot be safely represented by Q8_0".into());
        }
    }
    Ok(())
}

fn write_q8(values: &[f32], output: &mut dyn Write) -> Result<()> {
    let byte_len = values.len() / 32 * 34;
    // The C block contains fp16 fields: a Vec<u8> does not promise alignment.
    let mut storage = vec![0_u16; byte_len / 2];
    // SAFETY: values is contiguous, finite, and prevalidated, length is a
    // multiple of 64 and <= CHUNK_ELEMENTS; destination is aligned and sized
    // for exactly len/32 Q8_0 blocks. Q8_0 needs no importance matrix.
    let written = unsafe {
        ggml::ggml_quantize_chunk(
            ggml::GGML_TYPE_Q8_0,
            values.as_ptr(),
            storage.as_mut_ptr().cast(),
            0,
            1,
            values.len() as i64,
            std::ptr::null(),
        )
    };
    if written != byte_len {
        return Err(format!(
            "GGML Q8_0 wrote {written} bytes, expected {byte_len}"
        ));
    }
    // SAFETY: all storage is initialized; u8 can view any u16 representation.
    let bytes = unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), byte_len) };
    output.write_all(bytes).map_err(io_error)
}

fn validate_scales(file: &mut File, mut remaining: u64) -> Result<()> {
    let mut bytes = vec![0; remaining.min(IO_CHUNK as u64) as usize];
    while remaining != 0 {
        let n = remaining.min(bytes.len() as u64) as usize;
        file.read_exact(&mut bytes[..n]).map_err(io_error)?;
        for &scale in &bytes[..n] {
            if scale & 0x7f == 0x7f {
                return Err("nonfinite E4M3 NVFP4 scale".into());
            }
            // Native UE4M3 ignores the sign. Signed scales cannot be repacked
            // losslessly with unchanged bytes, so reject rather than corrupt.
            if scale & 0x80 != 0 {
                return Err("signed E4M3 NVFP4 scales are unsupported".into());
            }
        }
        remaining -= n as u64;
    }
    Ok(())
}

fn elements(shape: &[u64]) -> Result<u64> {
    if shape.len() > 4 {
        return Err("tensor rank exceeds GGML's four dimensions".into());
    }
    let mut n = 1;
    for &dim in shape {
        if dim == 0 || dim > i64::MAX as u64 {
            return Err("tensor dimensions must be positive int64 values".into());
        }
        n = mul(n, dim)?;
    }
    if n > i64::MAX as u64 {
        return Err("tensor element count exceeds int64".into());
    }
    Ok(n)
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| "tensor offset overflow".into())
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| "tensor size overflow".into())
}
fn io_error(e: std::io::Error) -> String {
    format!("tensor I/O: {e}")
}
fn seek_tensor(file: &mut File, data_start: u64, desc: &Descriptor) -> Result<()> {
    file.seek(SeekFrom::Start(add(data_start, desc.data_offsets[0])?))
        .map_err(io_error)?;
    Ok(())
}
fn copy_exact(input: &mut File, output: &mut dyn Write, mut remaining: u64) -> Result<()> {
    let mut bytes = vec![0; remaining.min(IO_CHUNK as u64) as usize];
    while remaining != 0 {
        let n = remaining.min(bytes.len() as u64) as usize;
        input.read_exact(&mut bytes[..n]).map_err(io_error)?;
        output.write_all(&bytes[..n]).map_err(io_error)?;
        remaining -= n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn windows_file_id(file: &File) -> Result<(u32, u64)> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    #[derive(Default)]
    struct FileInformation {
        attributes: u32,
        creation_time: [u32; 2],
        last_access_time: [u32; 2],
        last_write_time: [u32; 2],
        volume_serial: u32,
        size_high: u32,
        size_low: u32,
        links: u32,
        index_high: u32,
        index_low: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileType(handle: *mut c_void) -> u32;
        fn GetFileInformationByHandle(handle: *mut c_void, info: *mut FileInformation) -> i32;
    }
    let mut info = FileInformation::default();
    // SAFETY: File owns a live handle; info matches BY_HANDLE_FILE_INFORMATION
    // and is writable for its full size. FILE_TYPE_DISK is 1; never read pipes
    // or character devices even if their attributes resemble a plain file.
    unsafe {
        if GetFileType(file.as_raw_handle()) != 1 {
            return Err("tensor source is not a disk file".into());
        }
        if GetFileInformationByHandle(file.as_raw_handle(), &mut info) == 0 {
            return Err(io_error(std::io::Error::last_os_error()));
        }
    }
    Ok((
        info.volume_serial,
        (u64::from(info.index_high) << 32) | u64::from(info.index_low),
    ))
}

// Walk via directory descriptors so no component can be swapped for a symlink
// between checking and opening. O_NONBLOCK prevents a malicious FIFO replacement
// from hanging before fstat can reject it. No libc dependency is necessary.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(super) fn open_regular(path: &Path) -> Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;
    #[cfg(target_os = "macos")]
    const FLAGS: (i32, i32, i32, i32) = (0x100, 0x4, 0x100000, 0x1000000);
    #[cfg(target_os = "linux")]
    const FLAGS: (i32, i32, i32, i32) = (0x20000, 0x800, 0x10000, 0x80000);
    let (nofollow, nonblock, directory, cloexec) = FLAGS;
    unsafe extern "C" {
        fn openat(fd: i32, path: *const std::ffi::c_char, flags: i32, ...) -> i32;
    }
    let mut names = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => names.push(name),
            Component::RootDir | Component::CurDir => {}
            _ => return Err("tensor path must not contain '..' or platform prefixes".into()),
        }
    }
    if names.is_empty() {
        return Err("tensor path must name a regular file".into());
    }
    // Reject known special files without opening them at all (some device
    // drivers ignore O_NONBLOCK). The descriptor walk below still closes the
    // symlink/FIFO race and verifies that the same regular file was opened.
    let before = std::fs::symlink_metadata(path).map_err(io_error)?;
    if !before.is_file() {
        return Err("tensor source is not a regular file".into());
    }
    let mut file = File::open(if path.is_absolute() { "/" } else { "." }).map_err(io_error)?;
    let last = names.len() - 1;
    for (i, name) in names.into_iter().enumerate() {
        let name = CString::new(name.as_bytes()).map_err(|_| "NUL in tensor path")?;
        let flags = nofollow | nonblock | cloexec | if i == last { 0 } else { directory };
        // SAFETY: live directory fd and NUL-terminated name; no O_CREAT, so the
        // variadic mode argument is not needed. Returned fd is owned exactly once.
        let fd = unsafe { openat(file.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io_error(std::io::Error::last_os_error()));
        }
        file = unsafe { File::from_raw_fd(fd) };
    }
    let after = file.metadata().map_err(io_error)?;
    if !after.is_file() || Source::of(&before) != Source::of(&after) {
        return Err("tensor source is not a regular file or changed while opening".into());
    }
    Ok(file)
}

// Unlike the Unix descriptor walk, these pathname checks are NOT atomic with
// opening children. Windows callers must trust parent directories not to be
// replaced/renamed or changed into reparse points during this operation. The
// final handle does not follow a reparse point and denies writers/deleters for
// its lifetime; repeated checks detect changes, but do not eliminate parent
// traversal races. Only local disk paths are accepted, not UNC/device namespaces.
#[cfg(windows)]
pub(super) fn open_regular(path: &Path) -> Result<File> {
    use std::fs::OpenOptions;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use std::path::{Component, PathBuf, Prefix};

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    const FILE_ATTRIBUTE_DEVICE: u32 = 0x40;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_SHARE_READ: u32 = 1;

    for component in path.components() {
        match component {
            Component::ParentDir => return Err("tensor path must not contain '..'".into()),
            Component::Prefix(prefix) => {
                if !path.is_absolute()
                    || !matches!(prefix.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_))
                {
                    return Err("tensor path must use a local absolute disk prefix".into());
                }
            }
            Component::Normal(name) => {
                if name.encode_wide().any(|c| c == u16::from(b':')) {
                    return Err("alternate data streams are not tensor files".into());
                }
            }
            _ => {}
        }
    }
    let absolute = std::path::absolute(path).map_err(io_error)?;
    let check_components = || -> Result<Metadata> {
        let mut current = PathBuf::new();
        let mut last = None;
        for component in absolute.components() {
            current.push(component.as_os_str());
            if matches!(component, Component::Prefix(_)) {
                continue;
            }
            let meta = std::fs::symlink_metadata(&current).map_err(io_error)?;
            if meta.file_attributes() & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE) != 0
            {
                return Err("tensor path contains a reparse point or device".into());
            }
            if current != absolute && !meta.is_dir() {
                return Err("tensor parent is not a directory".into());
            }
            last = Some(meta);
        }
        let meta = last.ok_or("tensor path must name a regular file")?;
        if !meta.is_file() {
            return Err("tensor source is not a regular file".into());
        }
        Ok(meta)
    };
    let before = check_components()?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .share_mode(FILE_SHARE_READ)
        .open(&absolute)
        .map_err(io_error)?;
    let after = file.metadata().map_err(io_error)?;
    if !after.is_file()
        || after.file_attributes() & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE) != 0
        || Source::of(&before) != Source::of(&after)
        || Source::of(&after) != Source::of(&check_components()?)
    {
        return Err("tensor source is not a regular file or changed while opening".into());
    }
    // Stable std exposes no pre-open Windows file ID. Compare its available
    // metadata above; retain the handle ID for subsequent inspect/write checks.
    windows_file_id(&file)?;
    Ok(file)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
pub(super) fn open_regular(_path: &Path) -> Result<File> {
    Err("secure tensor opening is supported on macOS, Linux, and Windows only".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::fs;

    const NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";

    fn fixture(header: &Value, data: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        raw_fixture(&serde_json::to_vec(header).unwrap(), data)
    }
    fn raw_fixture(header: &[u8], data: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        // macOS's system temp directory can itself have a symlink ancestor.
        let path = dir.path().canonicalize().unwrap().join("tensor");
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(data).unwrap();
        (dir, path)
    }
    fn nv_header(rows: u64, cols: u64) -> Value {
        let w = rows * cols / 2;
        let s = rows * cols / 16;
        json!({
            "__metadata__": {"quant_type": "nvfp4"},
            NAME: {"dtype": "U32", "shape": [rows, cols / 8], "data_offsets": [0, w]},
            format!("{NAME}.scale"): {"dtype": "U8", "shape": [rows, cols / 16], "data_offsets": [w, w+s]},
            format!("{NAME}.global_scale"): {"dtype": "F32", "shape": [], "data_offsets": [w+s, w+s+4]}
        })
    }
    fn nv_data(rows: usize, cols: usize, global: f32) -> Vec<u8> {
        let mut data: Vec<u8> = (0..rows * cols / 2)
            .map(|i| {
                let j = i * 2;
                ((j % 16) as u8) | ((((j + 1) % 16) as u8) << 4)
            })
            .collect();
        let scale_values = [0, 1, 7, 8, 0x38, 0x40, 0x7e, 0x29];
        data.extend((0..rows * cols / 16).map(|i| scale_values[i % scale_values.len()]));
        data.extend(global.to_le_bytes());
        data
    }
    fn write_bytes(t: &Tensor, path: &Path, q8: bool) -> Vec<u8> {
        let mut out = Vec::new();
        t.write(path, &mut out, q8).unwrap();
        assert_eq!(out.len() as u64, t.output_info(q8).1);
        out
    }

    // GGML's fp16 lookup table is initialized by a context, not by this codec.
    // Only the tests' native reference dequantizer needs it.
    fn native_dequant(kind: u32, bytes: &[u8], n: usize) -> Vec<f32> {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| unsafe {
            let ctx = ggml::ggml_init(ggml::ggml_init_params {
                mem_size: 0,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            assert!(!ctx.is_null());
            ggml::ggml_free(ctx);
        });
        let mut aligned = vec![0_u16; bytes.len().div_ceil(2)];
        // SAFETY: destination is initialized and at least bytes.len() bytes long.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                aligned.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        let mut values = vec![0.0; n];
        unsafe {
            let traits = ggml::ggml_get_type_traits(kind);
            assert!(!traits.is_null());
            ((*traits).to_float.unwrap())(aligned.as_ptr().cast(), values.as_mut_ptr(), n as i64);
        }
        values
    }

    #[test]
    fn native_repack_preserves_scales_and_matches_c_dequantization() {
        let (rows, cols) = (2, 128);
        let data = nv_data(rows, cols, 2.75);
        let (_dir, path) = fixture(&nv_header(rows as u64, cols as u64), &data);
        let tensor = inspect(&path, NAME).unwrap();
        assert_eq!(tensor.shape, [2, 128]);
        assert_eq!(tensor.output_info(false), (ggml::GGML_TYPE_NVFP4, 144));
        assert_eq!(tensor.global_scale, Some(2.75));
        let bytes = write_bytes(&tensor, &path, false);
        let weights_len = rows * cols / 2;
        for (i, block) in bytes.as_chunks::<36>().0.iter().enumerate() {
            assert_eq!(
                &block[..4],
                &data[weights_len + i * 4..weights_len + i * 4 + 4]
            );
            for sub in 0..4 {
                for j in 0..8 {
                    assert_eq!(block[4 + sub * 8 + j], j as u8 | (((j + 8) as u8) << 4));
                }
            }
        }
        let decoded = native_dequant(ggml::GGML_TYPE_NVFP4, &bytes, rows * cols);
        let reference = dequantize(
            &data[..weights_len],
            &data[weights_len..data.len() - 4],
            1.0,
        );
        assert_eq!(decoded, reference);
        // Check reference E2M1/E4M3 values independently of the codec helpers.
        assert_eq!(decoded[16 + 1], 0.5 / 512.0);
        assert_eq!(decoded[16 + 7], 6.0 / 512.0);
        assert_eq!(decoded[6 * 16 + 7], 6.0 * 448.0);
    }

    #[test]
    fn embedding_q8_folds_global_once_and_matches_numeric_reference() {
        let (rows, cols) = (3, 128);
        for global in [0.0, 0.125, -2.75] {
            let data = nv_data(rows, cols, global);
            let (_dir, path) = fixture(&nv_header(rows as u64, cols as u64), &data);
            let tensor = inspect(&path, NAME).unwrap();
            assert_eq!(tensor.output_info(true), (ggml::GGML_TYPE_Q8_0, 408));
            let bytes = write_bytes(&tensor, &path, true);
            let native = native_dequant(
                ggml::GGML_TYPE_NVFP4,
                &write_bytes(&tensor, &path, false),
                rows * cols,
            );
            let decoded = native_dequant(ggml::GGML_TYPE_Q8_0, &bytes, rows * cols);
            for ((block, reference), packed) in decoded
                .as_chunks::<32>()
                .0
                .iter()
                .zip(native.as_chunks::<32>().0)
                .zip(bytes.as_chunks::<34>().0)
            {
                let reference: Vec<f32> = reference.iter().map(|v| v * global).collect();
                let max = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                let d = max / 127.0;
                for (i, (&actual, &expected)) in block.iter().zip(&reference).enumerate() {
                    let q = packed[2 + i] as i8;
                    let expected_q = if d == 0.0 {
                        0
                    } else {
                        (expected * (1.0 / d)).round() as i8
                    };
                    assert_eq!(q, expected_q);
                    assert!((actual - expected).abs() <= d * 0.6 + max * 0.001 + 1e-6);
                }
            }
        }
    }

    #[test]
    fn plain_types_vectors_and_scalars_copy_bit_for_bit() {
        for (dtype, ty, width) in [
            ("BF16", ggml::GGML_TYPE_BF16, 2),
            ("F16", ggml::GGML_TYPE_F16, 2),
            ("F32", ggml::GGML_TYPE_F32, 4),
        ] {
            for shape in [vec![], vec![1], vec![3], vec![2, 3]] {
                let n = shape.iter().product::<usize>() * width;
                let data: Vec<u8> = (0..n).map(|i| (i * 73) as u8).collect();
                let header =
                    json!({NAME: {"dtype": dtype, "shape": shape, "data_offsets": [0, n]}});
                let (_dir, path) = fixture(&header, &data);
                let tensor = inspect(&path, NAME).unwrap();
                assert_eq!(tensor.ggml_type, ty);
                assert_eq!(
                    tensor.shape,
                    shape.iter().map(|&n| n as u64).collect::<Vec<_>>()
                );
                assert_eq!(write_bytes(&tensor, &path, false), data);
                assert_eq!(write_bytes(&tensor, &path, true), data);
            }
        }
    }

    #[test]
    fn malformed_headers_ranges_shapes_and_dtypes_are_rejected() {
        let data = nv_data(1, 64, 1.0);
        let good = nv_header(1, 64);
        let scale = format!("{NAME}.scale");
        let global = format!("{NAME}.global_scale");
        let mut cases = Vec::new();
        macro_rules! bad { ($($key:expr),+ => $($value:tt)+) => {{
            let mut h = good.clone(); h$([$key])+ = json!($($value)+); cases.push(h);
        }}; }
        bad!("__metadata__", "quant_type" => "other");
        bad!("__metadata__", "extra" => "other");
        bad!(NAME, "dtype" => "I32");
        bad!(NAME, "dtype" => "F32");
        bad!(NAME, "shape" => [0, 8]);
        bad!(NAME, "shape" => [1, 7]);
        bad!(NAME, "shape" => [1, 8, 1]);
        bad!(NAME, "shape" => [u64::MAX, 8]);
        bad!(NAME, "shape" => [i64::MAX as u64, i64::MAX as u64]);
        bad!(NAME, "shape" => [1, 1, 1, 1, 8]);
        bad!(NAME, "extra" => 1);
        bad!(NAME, "data_offsets" => [32, 0]);
        bad!(NAME, "data_offsets" => [0, u64::MAX]);
        bad!(NAME, "data_offsets" => [0, 31]);
        bad!(NAME, "data_offsets" => [0, 32, 33]);
        bad!(&scale, "dtype" => "F16");
        bad!(&scale, "shape" => [1, 5]);
        bad!(&scale, "data_offsets" => [31, 35]);
        bad!(&scale, "data_offsets" => [33, 37]);
        bad!(&global, "dtype" => "U32");
        bad!(&global, "shape" => [1]);
        bad!("unrecognized" => {"dtype": "F32", "shape": [], "data_offsets": [36, 40]});
        let mut h = good.clone();
        h.as_object_mut().unwrap().remove("__metadata__");
        cases.push(h);
        let mut h = good.clone();
        h.as_object_mut().unwrap().remove(&scale);
        cases.push(h);
        let mut h = good.clone();
        h.as_object_mut().unwrap().remove(&global);
        cases.push(h);
        for h in cases {
            let (_dir, path) = fixture(&h, &data);
            assert!(inspect(&path, NAME).is_err(), "accepted {h}");
        }
        for bytes in [
            &data[..data.len() - 1],
            &[data.clone(), vec![0]].concat()[..],
        ] {
            let (_dir, path) = fixture(&good, bytes);
            assert!(inspect(&path, NAME).is_err());
        }
        for raw in [
            b"{}".as_slice(),
            b"[]",
            b"{",
            b"{\"x\":null}",
            b"{\"__metadata__\":null}",
        ] {
            let (_dir, path) = raw_fixture(raw, &[]);
            assert!(inspect(&path, NAME).is_err());
        }
    }

    #[test]
    fn duplicate_keys_and_oversized_truncated_headers_are_rejected() {
        let desc = r#"{"dtype":"F32","shape":[],"data_offsets":[0,4]}"#;
        let raw = format!("{{\"{NAME}\":{desc},\"{NAME}\":{desc}}}");
        let (_dir, path) = raw_fixture(raw.as_bytes(), &[0; 4]);
        assert!(inspect(&path, NAME).unwrap_err().contains("duplicate"));
        let raw = format!(
            "{{\"{NAME}\":{{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}}}"
        );
        let (_dir, path) = raw_fixture(raw.as_bytes(), &[0; 4]);
        assert!(inspect(&path, NAME).unwrap_err().contains("duplicate"));
        for len in [0, HEADER_LIMIT + 1, u64::MAX, 512] {
            fs::write(&path, len.to_le_bytes()).unwrap();
            assert!(inspect(&path, NAME).is_err());
        }
        fs::write(&path, [0; 7]).unwrap();
        assert!(inspect(&path, NAME).is_err());
    }

    #[test]
    fn nonfinite_global_or_scales_and_signed_scales_are_rejected() {
        for global in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let (_dir, path) = fixture(&nv_header(1, 64), &nv_data(1, 64, global));
            assert!(inspect(&path, NAME).is_err());
        }
        for scale in [0x7f, 0xff, 0x80, 0xb8] {
            let mut data = nv_data(1, 64, 1.0);
            data[32] = scale;
            let (_dir, path) = fixture(&nv_header(1, 64), &data);
            assert!(inspect(&path, NAME).is_err());
        }
    }

    #[test]
    fn validation_failures_do_not_touch_output() {
        let (_dir, path) = fixture(&nv_header(1, 64), &nv_data(1, 64, 1.0));
        let t = inspect(&path, NAME).unwrap();
        let mut output = Vec::new();
        let mut data = nv_data(1, 64, 1.0);
        data[32] = 0x7f;
        let (_dir2, bad_path) = fixture(&nv_header(1, 64), &data);
        assert!(t.write(&bad_path, &mut output, false).is_err());
        assert!(output.is_empty());
        fs::write(&path, [0; 16]).unwrap();
        assert!(t.write(&path, &mut output, false).is_err());
        assert!(output.is_empty());
        // A bad value in the LAST row is caught by preflight before output.
        let mut data = nv_data(2, 64, f32::MAX);
        data[64..71].fill(0);
        data[71] = 0x7e;
        let (_dir, path) = fixture(&nv_header(2, 64), &data);
        let t = inspect(&path, NAME).unwrap();
        assert!(t.write(&path, &mut output, true).is_err());
        assert!(output.is_empty());
        // Native repacking does not fold an otherwise finite global scalar.
        assert_eq!(write_bytes(&t, &path, false).len(), 72);
    }

    #[test]
    fn every_finite_unsigned_scale_matches_native_c() {
        let cols = 128 * 16;
        let mut data = nv_data(1, cols, 1.0);
        for i in 0..128 {
            data[cols / 2 + i] = (i % 127) as u8;
        }
        let (_dir, path) = fixture(&nv_header(1, cols as u64), &data);
        let t = inspect(&path, NAME).unwrap();
        let decoded = native_dequant(ggml::GGML_TYPE_NVFP4, &write_bytes(&t, &path, false), cols);
        assert_eq!(
            decoded,
            dequantize(&data[..cols / 2], &data[cols / 2..data.len() - 4], 1.0)
        );
    }

    #[test]
    fn companion_offsets_need_not_follow_weight_offsets() {
        let original = nv_data(2, 128, 0.25);
        let mut header = nv_header(2, 128);
        header[NAME]["data_offsets"] = json!([20, 148]);
        header[format!("{NAME}.scale")]["data_offsets"] = json!([4, 20]);
        header[format!("{NAME}.global_scale")]["data_offsets"] = json!([0, 4]);
        let mut data = original[144..148].to_vec();
        data.extend_from_slice(&original[128..144]);
        data.extend_from_slice(&original[..128]);
        let (_dir, path) = fixture(&header, &data);
        let t = inspect(&path, NAME).unwrap();
        assert_eq!(
            write_bytes(&t, &path, false),
            repack(&original[..128], &original[128..144])
        );
        let expected = dequantize(&original[..128], &original[128..144], 0.25);
        let mut q8 = Vec::new();
        write_q8(&expected, &mut q8).unwrap();
        assert_eq!(write_bytes(&t, &path, true), q8);
    }

    #[test]
    fn chunk_and_row_boundaries_are_preserved() {
        let cols = CHUNK_ELEMENTS + 64;
        let data = nv_data(2, cols, 0.5);
        let (_dir, path) = fixture(&nv_header(2, cols as u64), &data);
        let t = inspect(&path, NAME).unwrap();
        let native = write_bytes(&t, &path, false);
        assert_eq!(native, repack(&data[..cols], &data[cols..data.len() - 4]));
        let q8 = write_bytes(&t, &path, true);
        let expected = dequantize(&data[..cols], &data[cols..data.len() - 4], 0.5);
        let mut reference = Vec::new();
        write_q8(&expected, &mut reference).unwrap();
        assert_eq!(q8, reference);
    }

    #[cfg(windows)]
    #[test]
    fn windows_rejects_special_paths_and_denies_writers_and_deleters() {
        let (dir, path) = fixture(&nv_header(1, 64), &nv_data(1, 64, 1.0));
        let reader = open_regular(&path).unwrap();
        assert!(open_regular(&path).is_ok());
        assert!(fs::OpenOptions::new().write(true).open(&path).is_err());
        assert!(fs::remove_file(&path).is_err());
        drop(reader);
        let writer = fs::OpenOptions::new().write(true).open(&path).unwrap();
        assert!(open_regular(&path).is_err());
        drop(writer);
        assert!(open_regular(&dir.path().canonicalize().unwrap()).is_err());
        assert!(open_regular(&path.with_extension("tensor:stream")).is_err());
        assert!(open_regular(Path::new(r"\\.\pipe\not-a-tensor")).is_err());
        assert!(open_regular(Path::new(r"\\server\share\tensor")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_directories_and_fifos_are_rejected_without_blocking() {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::symlink;
        let (dir, path) = fixture(&nv_header(1, 64), &nv_data(1, 64, 1.0));
        let root = dir.path().canonicalize().unwrap();
        let link = root.join("link");
        symlink(&path, &link).unwrap();
        assert!(inspect(&link, NAME).is_err());
        assert!(inspect(&root, NAME).is_err());
        let alias = root.join("alias");
        symlink(&root, &alias).unwrap();
        assert!(inspect(&alias.join("tensor"), NAME).is_err());
        let fifo = root.join("fifo");
        let cpath = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        #[cfg(target_os = "macos")]
        type Mode = u16;
        #[cfg(not(target_os = "macos"))]
        type Mode = u32;
        unsafe extern "C" {
            fn mkfifo(path: *const std::ffi::c_char, mode: Mode) -> i32;
        }
        assert_eq!(unsafe { mkfifo(cpath.as_ptr(), 0o600) }, 0);
        assert!(inspect(&fifo, NAME).is_err());
        assert!(inspect(Path::new("/dev/null"), NAME).is_err());
        assert!(inspect(&root.join("../tensor"), NAME).is_err());
    }
}
