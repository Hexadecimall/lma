//! Process-global native callbacks use static storage, never a borrowed UI pointer.
use super::text::sanitize;
use llama_cpp_sys_2 as sys;
use std::{
    collections::VecDeque,
    ffi::CStr,
    sync::{Mutex, OnceLock},
};

const MAX_ENTRIES: usize = 512;
const MAX_ENTRY: usize = 4096;
static BUFFER: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();

fn buffer() -> &'static Mutex<VecDeque<String>> {
    BUFFER.get_or_init(|| Mutex::new(VecDeque::new()))
}

unsafe extern "C" fn callback(
    level: sys::ggml_log_level,
    text: *const std::ffi::c_char,
    _: *mut std::ffi::c_void,
) {
    if text.is_null() {
        return;
    }
    // llama.cpp promises a NUL-terminated string for the duration of this call.
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_ENTRY)]);
    let entry = format!(
        "[{level}] {}{}",
        sanitize(&text),
        if bytes.len() > MAX_ENTRY {
            " [log entry truncated]"
        } else {
            ""
        }
    );
    if let Ok(mut logs) = buffer().lock() {
        if logs.len() == MAX_ENTRIES {
            logs.pop_front();
        }
        logs.push_back(entry);
    }
}

pub struct NativeLogs;
impl NativeLogs {
    /// Install before loading, and drop only after worker teardown.
    pub fn capture() -> Self {
        if let Ok(mut logs) = buffer().lock() {
            logs.clear();
        }
        unsafe {
            sys::llama_log_set(Some(callback), std::ptr::null_mut());
            sys::ggml_log_set(Some(callback), std::ptr::null_mut());
            #[cfg(feature = "vision")]
            sys::mtmd_helper_log_set(Some(callback), std::ptr::null_mut());
        }
        Self
    }
}
impl Drop for NativeLogs {
    fn drop(&mut self) {
        unsafe {
            sys::llama_log_set(None, std::ptr::null_mut());
            sys::ggml_log_set(None, std::ptr::null_mut());
            #[cfg(feature = "vision")]
            sys::mtmd_helper_log_set(None, std::ptr::null_mut());
        }
    }
}

pub fn snapshot() -> String {
    buffer()
        .lock()
        .map(|logs| logs.iter().cloned().collect::<Vec<_>>().join(""))
        .unwrap_or_else(|_| "Native log buffer unavailable".into())
}

pub fn recent() -> String {
    let all = snapshot();
    let lines: Vec<_> = all.lines().collect();
    lines[lines.len().saturating_sub(12)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn callback_retains_errors_sanitizes_and_bounds_memory() {
        let message = std::ffi::CString::new("native GPU error\x1b]52;c;payload\x07\n").unwrap();
        for _ in 0..MAX_ENTRIES + 10 {
            unsafe {
                callback(
                    sys::GGML_LOG_LEVEL_ERROR,
                    message.as_ptr(),
                    std::ptr::null_mut(),
                );
            }
        }
        let snapshot = snapshot();
        assert!(snapshot.contains("native GPU error"));
        assert!(!snapshot.contains("payload"));
        assert_eq!(buffer().lock().unwrap().len(), MAX_ENTRIES);
        assert!(snapshot.len() <= MAX_ENTRIES * (MAX_ENTRY + 64));
    }
}
