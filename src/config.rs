use std::path::PathBuf;

use serde::{Serialize, Serializer};

#[derive(Debug, Serialize)]
pub struct Plan {
    #[serde(flatten)]
    pub job: Job,
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "command", content = "options", rename_all = "lowercase")]
pub enum Job {
    Serve(ServeConfig),
    Stop(StopConfig),
    Quantize(QuantizeConfig),
    Fetch(FetchConfig),
    Image(ImageConfig),
    Launch(LaunchConfig),
    List(ListConfig),
    Import(ImportConfig),
    Create(CreateConfig),
    Delete(DeleteConfig),
    Transcribe(crate::speech::TranscribeConfig),
}

#[derive(Clone, Debug, Serialize)]
pub struct ServeConfig {
    pub model: PathBuf,
    pub model_id: Option<String>,
    pub ollama_dir: Option<PathBuf>,
    pub projector: Option<PathBuf>,
    pub chat_template: String,
    /// Terminal reply cap: None keeps the default, Some(0) uses remaining context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<crate::models::modelfile::Definition>,
    /// Modelfile load keys explicitly set by the caller, including `template`.
    #[serde(skip)]
    pub explicit_load: Vec<String>,
    pub prompt_cache: bool,
    pub kv_cache: String,
    pub memory_fallback: String,
    /// Compact routed-expert device-slot budget in MiB; zero disables it.
    pub expert_cache_mib: u32,
    pub host: String,
    pub port: u16,
    pub api: String,
    pub client: String,
    pub gpu_layers: u32,
    pub context: u32,
    pub threads: i32,
    pub batch: u32,
    pub micro_batch: u32,
    pub flash: String,
    #[serde(serialize_with = "redact_key")]
    pub api_key: Option<String>,
}

impl ServeConfig {
    /// Expert caching preserves requested batching. GPU slots serve decode and
    /// SILU prefill of 2..=16 local tokens; larger/empty views use mapped CPU experts.
    /// Explicit CPU caching is unchanged; graph metadata/workspaces are outside
    /// the slot budget and can grow for worst-case 16-token stream expansion.
    pub(crate) fn apply_expert_cache_policy(&mut self) -> Result<(), String> {
        if self.expert_cache_mib == 0 {
            return Ok(());
        }
        if self.expert_cache_mib > 1_048_576 {
            return Err("expert_cache_mib must be at most 1048576 MiB".into());
        }
        if self.projector.is_some() {
            return Err(
                "expert caching supports text generation only; remove the vision projector".into(),
            );
        }

        Ok(())
    }
}

/// Zero is an internal terminal setting, never sent as an HTTP max_tokens value.
pub fn chat_token_limit(value: &str) -> Result<u32, String> {
    match value {
        "off" | "unlimited" | "0" => Ok(0),
        _ => value.parse::<u32>().ok().filter(|n| *n > 0)
            .ok_or_else(|| "reply tokens must be a positive integer, or off (0) for remaining-context generation".into()),
    }
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            model_id: None,
            ollama_dir: None,
            projector: None,
            chat_template: "auto".into(),
            chat_tokens: None,
            definition: None,
            explicit_load: Vec::new(),
            prompt_cache: true,
            kv_cache: "f16".into(),
            memory_fallback: "error".into(),
            expert_cache_mib: 0,
            host: "127.0.0.1".into(),
            port: 8080,
            api: "all".into(),
            client: "default".into(),
            gpu_layers: default_gpu_layers(),
            context: 4096,
            threads: default_threads(),
            batch: 512,
            micro_batch: 128,
            flash: "auto".into(),
            api_key: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StopConfig {
    pub model: Option<String>,
    pub host: String,
    pub port: u16,
    #[serde(serialize_with = "redact_key")]
    pub api_key: Option<String>,
}

impl Default for StopConfig {
    fn default() -> Self {
        Self {
            model: None,
            host: "127.0.0.1".into(),
            port: 8080,
            api_key: None,
        }
    }
}

/// Formatting choices supported by the pinned llama.cpp renderer, plus lma's
/// automatic policy, unified Gemma4 formatter, and plain-text base-model fallback.
pub const CHAT_TEMPLATES: &[&str] = &[
    "auto",
    "native",
    "plain",
    "chatml",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "llama3",
    "llama4",
    "gemma",
    "gemma4",
    "phi3",
    "phi4",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "falcon3",
    "zephyr",
    "monarch",
    "orion",
    "openchat",
    "vicuna",
    "vicuna-orca",
    "deepseek",
    "deepseek2",
    "deepseek3",
    "deepseek-ocr",
    "command-r",
    "chatglm3",
    "chatglm4",
    "glmedge",
    "minicpm",
    "exaone3",
    "exaone4",
    "exaone-moe",
    "rwkv-world",
    "granite",
    "granite-4.0",
    "granite-4.1",
    "gigachat",
    "megrez",
    "yandex",
    "bailing",
    "bailing-think",
    "bailing2",
    "smolvlm",
    "hunyuan-moe",
    "gpt-oss",
    "hunyuan-dense",
    "hunyuan-vl",
    "kimi-k2",
    "seed_oss",
    "grok-2",
    "pangu-embedded",
    "solar-open",
];

/// Connection presets only; none launch clients or modify their configuration.
pub const CLIENT_PRESETS: &[&str] = &[
    "default",
    "claude-code",
    "codex",
    "zed",
    "vscode",
    "opencode",
    "custom",
];

/// Used only when the CLI did not receive an explicit API selection.
pub fn default_api_for_client(client: &str) -> &'static str {
    match client {
        "claude-code" => "anthropic",
        "codex" | "zed" | "vscode" | "opencode" => "openai",
        _ => "all",
    }
}

/// An explicit launch destination, independent of print-only hosting presets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LaunchTarget {
    #[default]
    Chat,
    Opencode,
    Codex,
    Claude,
    OpencodeDesktop,
    CodexDesktop,
    ClaudeDesktop,
}

impl LaunchTarget {
    pub const ALL: [Self; 7] = [
        Self::Chat,
        Self::Opencode,
        Self::Codex,
        Self::Claude,
        Self::OpencodeDesktop,
        Self::CodexDesktop,
        Self::ClaudeDesktop,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Opencode => "opencode",
            Self::Codex => "codex",
            Self::Claude => "claude",
            Self::OpencodeDesktop => "opencode-desktop",
            Self::CodexDesktop => "codex-desktop",
            Self::ClaudeDesktop => "claude-desktop",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Chat => "Terminal chat",
            Self::Opencode => "OpenCode",
            Self::Codex => "Codex",
            Self::Claude => "Claude Code",
            Self::OpencodeDesktop => "OpenCode Desktop",
            Self::CodexDesktop => "Codex Desktop",
            Self::ClaudeDesktop => "Claude Desktop",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|target| target.as_str() == value)
            .ok_or_else(|| {
                "-tt must be one of: chat, opencode, codex, claude, opencode-desktop, codex-desktop, claude-desktop".into()
            })
    }

    pub const fn is_desktop(self) -> bool {
        matches!(
            self,
            Self::OpencodeDesktop | Self::CodexDesktop | Self::ClaudeDesktop
        )
    }

    pub const fn client_preset(self) -> Option<&'static str> {
        match self {
            Self::Chat => None,
            Self::Opencode | Self::OpencodeDesktop => Some("opencode"),
            Self::Codex | Self::CodexDesktop => Some("codex"),
            Self::Claude | Self::ClaudeDesktop => Some("claude-code"),
        }
    }
}

/// Shared setup for the interactive launcher, terminal chat, and hosted APIs.
#[derive(Clone, Debug, Serialize)]
pub struct LaunchConfig {
    #[serde(flatten)]
    pub serve: ServeConfig,
    /// Internal setup mode: launch, chat, or host. Hosting CLI uses `serve`.
    pub mode: String,
    pub target: LaunchTarget,
    pub interactive: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeleteConfig {
    pub model: String,
    pub source: Option<String>,
    pub ollama_dir: Option<PathBuf>,
    pub yes: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CreateConfig {
    pub model: String,
    pub file: PathBuf,
    pub ollama_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImportConfig {
    pub model: String,
    pub ollama_dir: Option<PathBuf>,
    pub output: PathBuf,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ListConfig {
    pub local_dirs: Vec<PathBuf>,
    pub ollama_dir: Option<PathBuf>,
    pub filter: String,
    pub json: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantizeConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub kind: String,
    pub threads: i32,
    pub pure: bool,
    pub leave_output: bool,
    pub allow_requantize: bool,
    pub force: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FetchConfig {
    pub query: String,
    pub ollama_database: bool,
    pub lm_studio_database: bool,
    pub list: bool,
    pub file: Option<String>,
    pub output: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImageConfig {
    pub model_dir: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub prompt: String,
    pub negative_prompt: String,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f64,
    pub device: String,
    pub attention_slicing: Option<usize>,
}

pub const QUANT_TYPES: &[&str] = &[
    "F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K_S", "Q3_K_M",
    "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
];

/// Matches the CLI's `--gpu all` policy; this is not a memory-fit guarantee.
pub const ALL_GPU_LAYERS: u32 = 999;

const fn metal_backend_compiled() -> bool {
    // Pinned llama-cpp-2 0.1.160 unconditionally enables sys/metal through its
    // Apple Silicon target dependency, even with default-features = false.
    cfg!(any(
        feature = "metal",
        all(target_os = "macos", target_arch = "aarch64")
    ))
}

pub const fn default_gpu_layers() -> u32 {
    if metal_backend_compiled() || cfg!(feature = "cuda") {
        ALL_GPU_LAYERS
    } else {
        0
    }
}

/// Build/request information only. Actual placement requires a loaded model and
/// native offload/buffer diagnostics; a compiled feature is not a detected GPU.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct BackendReport {
    pub compiled_backends: Vec<&'static str>,
    pub requested_backend: &'static str,
    pub requested_gpu_layers: u32,
    pub actual_backend: Option<&'static str>,
}

pub fn backend_report(gpu_layers: u32) -> BackendReport {
    let mut compiled_backends = vec!["cpu"];
    if metal_backend_compiled() {
        compiled_backends.push("metal");
    }
    if cfg!(feature = "cuda") {
        compiled_backends.push("cuda");
    }
    BackendReport {
        compiled_backends,
        requested_backend: if gpu_layers == 0 { "cpu" } else { "gpu" },
        requested_gpu_layers: gpu_layers,
        actual_backend: None,
    }
}

pub fn default_threads() -> i32 {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(i32::MAX as usize) as i32)
}

fn redact_key<S: Serializer>(key: &Option<String>, serializer: S) -> Result<S::Ok, S::Error> {
    key.as_ref().map(|_| "[redacted]").serialize(serializer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_targets_round_trip_serialize_and_map_to_client_presets() {
        assert_eq!(LaunchTarget::default(), LaunchTarget::Chat);
        for (target, name, desktop, client) in [
            (LaunchTarget::Chat, "chat", false, None),
            (LaunchTarget::Opencode, "opencode", false, Some("opencode")),
            (LaunchTarget::Codex, "codex", false, Some("codex")),
            (LaunchTarget::Claude, "claude", false, Some("claude-code")),
            (
                LaunchTarget::OpencodeDesktop,
                "opencode-desktop",
                true,
                Some("opencode"),
            ),
            (
                LaunchTarget::CodexDesktop,
                "codex-desktop",
                true,
                Some("codex"),
            ),
            (
                LaunchTarget::ClaudeDesktop,
                "claude-desktop",
                true,
                Some("claude-code"),
            ),
        ] {
            assert_eq!(LaunchTarget::parse(name).unwrap(), target);
            assert_eq!(target.as_str(), name);
            assert_eq!(serde_json::to_value(target).unwrap(), name);
            assert_eq!(target.is_desktop(), desktop);
            assert_eq!(target.client_preset(), client);
            assert!(!target.label().is_empty());
        }
        assert_eq!(LaunchTarget::ALL.len(), 7);
        for invalid in [
            "",
            "Chat",
            "claude-code",
            "desktop",
            "codex;echo nope",
            " codex",
        ] {
            assert!(LaunchTarget::parse(invalid).is_err());
        }
    }

    #[test]
    fn default_gpu_policy_matches_compiled_features() {
        let expected = if cfg!(any(
            feature = "metal",
            feature = "cuda",
            all(target_os = "macos", target_arch = "aarch64")
        )) {
            ALL_GPU_LAYERS
        } else {
            0
        };
        assert_eq!(default_gpu_layers(), expected);
        assert_eq!(ServeConfig::default().gpu_layers, expected);
        assert_eq!(ServeConfig::default().memory_fallback, "error");
        assert_eq!(ServeConfig::default().expert_cache_mib, 0);
    }

    #[test]
    fn backend_report_never_confuses_compiled_requested_and_actual() {
        for layers in [0, 1, ALL_GPU_LAYERS] {
            let report = backend_report(layers);
            assert!(report.compiled_backends.contains(&"cpu"));
            assert_eq!(
                report.compiled_backends.contains(&"metal"),
                cfg!(any(
                    feature = "metal",
                    all(target_os = "macos", target_arch = "aarch64")
                ))
            );
            assert_eq!(
                report.compiled_backends.contains(&"cuda"),
                cfg!(feature = "cuda")
            );
            assert_eq!(report.requested_gpu_layers, layers);
            assert_eq!(
                report.requested_backend,
                if layers == 0 { "cpu" } else { "gpu" }
            );
            assert_eq!(report.actual_backend, None);
            let json = serde_json::to_value(&report).unwrap();
            assert!(json["actual_backend"].is_null());
        }
    }

    #[test]
    fn model_definitions_serialize_but_explicit_load_bookkeeping_does_not() {
        let mut config = ServeConfig::default();
        assert!(config.definition.is_none());
        assert!(config.explicit_load.is_empty());
        let json = serde_json::to_value(&config).unwrap();
        assert!(json.get("definition").is_none());
        assert!(json.get("explicit_load").is_none());
        config.definition = Some(crate::models::modelfile::Definition {
            from: "tiny:latest".into(),
            system: Some("Be concise.".into()),
            ..Default::default()
        });
        config.explicit_load = vec!["num_ctx".into(), "template".into()];
        config.api_key = Some("definition-preview-secret".into());
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["definition"]["from"], "tiny:latest");
        assert_eq!(json["definition"]["system"], "Be concise.");
        assert!(json.get("explicit_load").is_none());
        assert_eq!(json["api_key"], "[redacted]");
        assert!(!json.to_string().contains("definition-preview-secret"));
    }

    #[test]
    fn preview_does_not_expose_api_keys() {
        let config = ServeConfig {
            api_key: Some("super-secret".into()),
            ..ServeConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("super-secret"));
        assert!(json.contains("[redacted]"));
        let stop = StopConfig {
            api_key: Some("stop-secret".into()),
            ..StopConfig::default()
        };
        let json = serde_json::to_string(&stop).unwrap();
        assert!(!json.contains("stop-secret"));
        assert!(json.contains("[redacted]"));
    }
}
