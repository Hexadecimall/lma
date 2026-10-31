//! Bounded GGUF v3 header construction; tensor payloads are streamed separately.
use std::{collections::BTreeSet, io::Write};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
    U32s(Vec<u32>),
    I32s(Vec<i32>),
    F32s(Vec<f32>),
    Bools(Vec<bool>),
    Strings(Vec<String>),
}

pub(super) struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub kind: u32,
    pub size: u64,
    pub offset: u64,
}

pub(super) struct Layout {
    header: Vec<u8>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
    pub total_bytes: u64,
}

const ALIGNMENT: u64 = 32;
const HEADER_LIMIT: usize = 128 * 1024 * 1024;

fn align(value: u64) -> Result<u64, String> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|n| n / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| "GGUF size overflow".into())
}
fn string(output: &mut Vec<u8>, value: &str) -> Result<(), String> {
    if value.len() > 64 * 1024 * 1024
        || output.len().saturating_add(value.len()).saturating_add(8) > HEADER_LIMIT
    {
        return Err("GGUF header exceeds 128 MiB".into());
    }
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}
fn array(output: &mut Vec<u8>, kind: u32, length: usize) {
    output.extend_from_slice(&kind.to_le_bytes());
    output.extend_from_slice(&(length as u64).to_le_bytes());
}
impl Value {
    fn encode(&self, output: &mut Vec<u8>) -> Result<(), String> {
        let kind: u32 = match self {
            Self::U32(_) => 4,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            _ => 9,
        };
        output.extend_from_slice(&kind.to_le_bytes());
        match self {
            Self::U32(value) => output.extend_from_slice(&value.to_le_bytes()),
            Self::F32(value) => {
                if !value.is_finite() {
                    return Err("nonfinite GGUF metadata".into());
                }
                output.extend_from_slice(&value.to_le_bytes());
            }
            Self::Bool(value) => output.push(u8::from(*value)),
            Self::String(value) => string(output, value)?,
            Self::U32s(values) => {
                array(output, 4, values.len());
                for value in values {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::I32s(values) => {
                array(output, 5, values.len());
                for value in values {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::F32s(values) => {
                array(output, 6, values.len());
                for value in values {
                    if !value.is_finite() {
                        return Err("nonfinite GGUF metadata".into());
                    }
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Bools(values) => {
                array(output, 7, values.len());
                output.extend(values.iter().map(|v| u8::from(*v)));
            }
            Self::Strings(values) => {
                array(output, 8, values.len());
                for value in values {
                    string(output, value)?;
                }
            }
        }
        if output.len() > HEADER_LIMIT {
            return Err("GGUF header exceeds 128 MiB".into());
        }
        Ok(())
    }
}
impl Layout {
    pub fn new(
        mut metadata: Vec<(String, Value)>,
        mut tensors: Vec<TensorInfo>,
    ) -> Result<Self, String> {
        if metadata.len() > 1024 || tensors.is_empty() || tensors.len() > 16384 {
            return Err("invalid GGUF metadata/tensor count".into());
        }
        metadata.push(("general.alignment".into(), Value::U32(ALIGNMENT as u32)));
        let mut names = BTreeSet::new();
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            if key.is_empty() || key.len() > 256 || key.contains('\0') || !names.insert(key.clone())
            {
                return Err("invalid or duplicate GGUF metadata key".into());
            }
            string(&mut header, &key)?;
            value.encode(&mut header)?;
        }
        names.clear();
        let mut data_size = 0u64;
        for tensor in &mut tensors {
            if tensor.name.is_empty()
                || tensor.name.len() > 127
                || tensor.name.contains('\0')
                || !names.insert(tensor.name.clone())
                || tensor.dimensions.is_empty()
                || tensor.dimensions.len() > 4
                || tensor.dimensions.contains(&0)
                || tensor.size == 0
            {
                return Err("invalid or duplicate GGUF tensor descriptor".into());
            }
            tensor.offset = align(data_size)?;
            data_size = tensor
                .offset
                .checked_add(tensor.size)
                .ok_or("GGUF tensor size overflow")?;
            string(&mut header, &tensor.name)?;
            header.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
            for dim in &tensor.dimensions {
                header.extend_from_slice(&dim.to_le_bytes());
            }
            header.extend_from_slice(&tensor.kind.to_le_bytes());
            header.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let data_start = align(header.len() as u64)?;
        let total_bytes = data_start
            .checked_add(data_size)
            .ok_or("GGUF file size overflow")?;
        Ok(Self {
            header,
            tensors,
            data_start,
            total_bytes,
        })
    }
    pub fn write_header(&self, output: &mut dyn Write) -> Result<(), String> {
        output.write_all(&self.header).map_err(|e| e.to_string())?;
        padding(output, self.data_start - self.header.len() as u64)
    }
}
pub(super) fn padding(output: &mut dyn Write, count: u64) -> Result<(), String> {
    if count >= ALIGNMENT {
        return Err("invalid GGUF alignment gap".into());
    }
    output
        .write_all(&[0; ALIGNMENT as usize][..count as usize])
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_reader_accepts_header_and_alignment() {
        let metadata = vec![(
            "general.architecture".into(),
            Value::String("gemma4".into()),
        )];
        let specs = vec![
            TensorInfo {
                name: "a".into(),
                dimensions: vec![1],
                kind: 0,
                size: 4,
                offset: 0,
            },
            TensorInfo {
                name: "b".into(),
                dimensions: vec![2],
                kind: 0,
                size: 8,
                offset: 0,
            },
        ];
        let layout = Layout::new(metadata, specs).unwrap();
        assert_eq!(layout.tensors[1].offset, 32);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        layout.write_header(file.as_file_mut()).unwrap();
        file.write_all(&[0; 40]).unwrap();
        assert_eq!(file.as_file().metadata().unwrap().len(), layout.total_bytes);
        let name = std::ffi::CString::new(file.path().to_str().unwrap()).unwrap();
        let params = llama_cpp_sys_2::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: live path/params; no allocation of tensor payloads is requested.
        let context = unsafe { llama_cpp_sys_2::gguf_init_from_file(name.as_ptr(), params) };
        assert!(!context.is_null());
        unsafe {
            assert_eq!(llama_cpp_sys_2::gguf_get_n_tensors(context), 2);
            llama_cpp_sys_2::gguf_free(context);
        }
    }

    #[test]
    fn duplicate_keys_names_and_nonfinite_values_are_rejected() {
        let spec = || TensorInfo {
            name: "a".into(),
            dimensions: vec![1],
            kind: 0,
            size: 4,
            offset: 0,
        };
        assert!(
            Layout::new(
                vec![("general.alignment".into(), Value::U32(32))],
                vec![spec()]
            )
            .is_err()
        );
        assert!(Layout::new(vec![], vec![spec(), spec()]).is_err());
        assert!(Layout::new(vec![("bad".into(), Value::F32(f32::NAN))], vec![spec()]).is_err());
    }
}
