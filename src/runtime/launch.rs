//! Owned CLI sessions backed by an ephemeral, loopback-only native lma host.
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    fs,
    io::Read,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use crate::config::{LaunchTarget, ServeConfig};
use serde_json::{Value, json};
use tokio::{sync::oneshot, task::JoinHandle};

const POLL: Duration = Duration::from_millis(50);
const VERSION_TIMEOUT: Duration = Duration::from_secs(3);
const HTTP_DRAIN: Duration = Duration::from_secs(3);
const MAX_CONFIG: u64 = 1024 * 1024;
static NEXT_PROVIDER: AtomicU64 = AtomicU64::new(0);

/// Preflight never opens Desktop, executes an installer, or loads model weights.
/// Chat is dispatched by the TUI, not through this external-client supervisor.
pub fn run(mut config: ServeConfig, target: LaunchTarget) -> Result<(), String> {
    let name = cli_name(target)?;
    let address = launch_address(&config)?;
    validate_key(&config)?;
    preflight_managed(name)?;
    let executable = preflight_executable(name)?;
    if name == "opencode" {
        preflight_opencode_config()?;
    }
    // Reserve the actual socket, not a probe that releases it before model loading.
    let listener = reserve_endpoint(address)?;
    let state = if name == "opencode" {
        Some(
            tempfile::Builder::new()
                .prefix("lma-opencode-")
                .tempdir()
                .map_err(|error| {
                    format!(
                        "cannot create private OpenCode runtime state: {error}; no model loaded"
                    )
                })?,
        )
    } else {
        None
    };
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    config.host = address.ip().to_string();
    config.port = address.port();
    config.client = if name == "claude" {
        "claude-code"
    } else {
        name
    }
    .into();
    // These routes belong to this launch, independent of a previously selected preset.
    config.api = if name == "claude" {
        "anthropic"
    } else {
        "openai"
    }
    .into();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot start client supervisor: {error}"))?
        .block_on(run_async(config, name, executable, listener, state))
}

fn cli_name(target: LaunchTarget) -> Result<&'static str, String> {
    match target {
        LaunchTarget::Opencode => Ok("opencode"),
        LaunchTarget::Codex => Ok("codex"),
        LaunchTarget::Claude => Ok("claude"),
        LaunchTarget::Chat => {
            Err("chat uses lma's built-in terminal chat, not an external client".into())
        }
        LaunchTarget::OpencodeDesktop
        | LaunchTarget::CodexDesktop
        | LaunchTarget::ClaudeDesktop => {
            let detail = match target {
                LaunchTarget::OpencodeDesktop => {
                    "OpenCode Desktop may adopt an existing shared server; configure and explicitly connect a dedicated server instead"
                }
                LaunchTarget::CodexDesktop => {
                    "codex app opens a workspace but does not forward CLI provider overrides; documented desktop provider setup requires configuration and restart"
                }
                _ => {
                    "clarify whether Claude's Code tab or regular Chat is intended; both require Desktop third-party inference setup, which writes configuration and relaunches"
                }
            };
            Err(format!(
                "desktop local-provider setup required, not opened; CLI overrides do not configure desktop; {detail}"
            ))
        }
    }
}

fn launch_address(config: &ServeConfig) -> Result<SocketAddr, String> {
    let host = config
        .host
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(&config.host);
    let ip = if host == "localhost" {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        host.parse::<IpAddr>().map_err(|_| "client launch requires localhost or a literal loopback IP; DNS and external binds are not allowed".to_string())?
    };
    if !ip.is_loopback() {
        return Err("client launch requires a loopback endpoint; external and wildcard binds are not allowed".into());
    }
    if config.context == 0 {
        return Err("client launch context must be greater than zero".into());
    }
    Ok(SocketAddr::new(ip, config.port))
}

fn reserve_endpoint(address: SocketAddr) -> Result<std::net::TcpListener, String> {
    let listener = std::net::TcpListener::bind(address).map_err(|error| {
        format!("cannot reserve local client endpoint {address}: {error}; no model loaded")
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|error| error.to_string())?;
    Ok(listener)
}

fn validate_key(config: &ServeConfig) -> Result<(), String> {
    if config.api_key.as_ref().is_some_and(|key| {
        key.is_empty() || key.chars().any(|c| c.is_whitespace() || c.is_control())
    }) {
        return Err(
            "api_key must be nonempty and contain no whitespace or control characters".into(),
        );
    }
    Ok(())
}

async fn run_async(
    config: ServeConfig,
    name: &'static str,
    executable: PathBuf,
    listener: std::net::TcpListener,
    state: Option<tempfile::TempDir>,
) -> Result<(), String> {
    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);
    // Register the handler before a potentially long native load. Never cancel a
    // partially completed Host::new: its blocking loader must finish ownership transfer.
    tokio::select! {
        biased;
        result = &mut interrupt => return signal_result(result),
        _ = std::future::ready(()) => {}
    }
    let config = tokio::task::spawn_blocking(move || {
        let mut config = config;
        let public_id = config.model_id.clone();
        super::resolve_model(&mut config)?;
        if public_id.is_some() {
            config.model_id = public_id;
        }
        Ok::<_, String>(config)
    })
    .await
    .map_err(|error| format!("model resolution task failed: {error}"))??;
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    // Build and validate the complete profile before calling the native loader.
    let mut plan = ClientPlan::new(name, executable, &config, address)?;
    if let Some(state) = state {
        // Keep user/project permission configuration, but do not import saved
        // provider accounts or touch the already-running client's state.
        plan.set("XDG_DATA_HOME", state.path().join("data"));
        plan.set("XDG_STATE_HOME", state.path().join("state"));
        plan.state = Some(state);
    }
    let listener =
        tokio::net::TcpListener::from_std(listener).map_err(|error| error.to_string())?;
    let (router, host) = super::hosting::Host::new(config).await?;
    let result = async {
        tokio::select! {
            biased;
            result = &mut interrupt => { drop(router); return signal_result(result); }
            _ = std::future::ready(()) => {}
        }
        eprintln!("lma local endpoint: http://{address}; starting {name} (client controls tool permissions)");
        let (http_stop, http_done) = oneshot::channel();
        let http = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async { let _ = http_done.await; })
                .await
                .map_err(|error| format!("local client HTTP server failed: {error}"))
        });
        let (child_stop, stop_requests) = mpsc::channel();
        let child = tokio::task::spawn_blocking(move || monitor_child(plan, stop_requests));
        supervise(child, child_stop, http, http_stop, &mut interrupt).await
    }.await;
    // supervise has drained or aborted and awaited the router task first.
    let shutdown = host.shutdown().await;
    combine(result, shutdown)
}

fn signal_result(result: std::io::Result<()>) -> Result<(), String> {
    result.map_err(|error| format!("cannot monitor Ctrl-C: {error}"))
}

async fn supervise(
    mut child: JoinHandle<Result<(), String>>,
    child_stop: mpsc::Sender<()>,
    mut http: JoinHandle<Result<(), String>>,
    http_stop: oneshot::Sender<()>,
    interrupt: impl std::future::Future<Output = std::io::Result<()>>,
) -> Result<(), String> {
    enum Finished {
        Child,
        Http,
        Signal,
    }
    let (finished, mut result) = tokio::select! {
        result = &mut child => (Finished::Child, joined(result, "client supervisor")),
        result = &mut http => (Finished::Http, joined(result, "HTTP server").and_then(|()| Err("local client HTTP server stopped before the client finished".into()))),
        result = interrupt => (Finished::Signal, signal_result(result)),
    };
    // Closing either sender on cancellation also requests owned-resource cleanup.
    let _ = child_stop.send(());
    drop(child_stop);
    if !matches!(finished, Finished::Child) {
        let cleanup = joined(child.await, "client supervisor");
        // A child killed because of Ctrl-C/server failure is expected; monitor_child
        // returns Ok for an intentional stop, but still reports kill/reap failures.
        result = combine(result, cleanup);
    }
    let _ = http_stop.send(());
    if !matches!(finished, Finished::Http) {
        match tokio::time::timeout(HTTP_DRAIN, &mut http).await {
            Ok(done) => result = combine(result, joined(done, "HTTP server")),
            Err(_) => {
                http.abort();
                let _ = http.await;
                // Active keep-alive/streaming connections cannot hold the host forever.
                eprintln!(
                    "lma: closed remaining client HTTP connections after shutdown grace period"
                );
            }
        }
    }
    result
}

fn joined(
    result: Result<Result<(), String>, tokio::task::JoinError>,
    task: &str,
) -> Result<(), String> {
    result.map_err(|error| format!("{task} task failed: {error}"))?
}

fn combine(first: Result<(), String>, cleanup: Result<(), String>) -> Result<(), String> {
    match (first, cleanup) {
        (Err(first), Err(cleanup)) => Err(format!("{first}; cleanup failed: {cleanup}")),
        (Err(error), _) | (_, Err(error)) => Err(error),
        _ => Ok(()),
    }
}

// This type deliberately has no Debug implementation: env contains credentials.
struct ClientPlan {
    executable: PathBuf,
    args: Vec<OsString>,
    env: BTreeMap<OsString, OsString>,
    remove: Vec<OsString>,
    name: &'static str,
    state: Option<tempfile::TempDir>,
}

impl ClientPlan {
    fn new(
        name: &'static str,
        executable: PathBuf,
        config: &ServeConfig,
        address: SocketAddr,
    ) -> Result<Self, String> {
        let inline = if name == "opencode" {
            inline_opencode()?
        } else {
            json!({})
        };
        Self::with_inline(name, executable, config, address, inline)
    }

    fn with_inline(
        name: &'static str,
        executable: PathBuf,
        config: &ServeConfig,
        address: SocketAddr,
        inline: Value,
    ) -> Result<Self, String> {
        let model = config
            .model_id
            .clone()
            .or_else(|| {
                config
                    .model
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
            })
            .filter(|s| !s.is_empty() && !s.contains('\0'))
            .ok_or_else(|| {
                "client launch requires a nonempty model ID without NUL characters".to_string()
            })?;
        let root = format!("http://{address}");
        let base = format!("{root}/v1");
        let key = config.api_key.as_deref().unwrap_or("local-not-required");
        let output = (config.context / 4).clamp(1, 256);
        let mut plan = Self {
            executable,
            args: Vec::new(),
            env: BTreeMap::new(),
            remove: Vec::new(),
            name,
            state: None,
        };
        // A shell proxy must not receive the local inference payload or credentials.
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ] {
            plan.remove.push(key.into());
        }
        plan.set("NO_PROXY", "127.0.0.1,localhost,::1");
        plan.set("no_proxy", "127.0.0.1,localhost,::1");
        match name {
            "codex" => {
                // Never attach this launch to an existing app-server daemon.
                plan.args.push("--no-daemon".into());
                let provider = format!(
                    "lma_launch_{}_{}",
                    std::process::id(),
                    NEXT_PROVIDER.fetch_add(1, Ordering::Relaxed)
                );
                for value in [
                    format!("model={}", quoted(&model)),
                    format!("model_provider={}", quoted(&provider)),
                    format!("model_context_window={}", config.context),
                    "model_supports_reasoning_summaries=false".into(),
                    "web_search=\"disabled\"".into(),
                    "features.enable_request_compression=false".into(),
                    format!(
                        "model_providers.{provider}={{ name=\"lma\", base_url={}, wire_api=\"responses\", env_key=\"LMA_API_KEY\", requires_openai_auth=false, supports_websockets=false, capabilities={{remote_compaction=\"unsupported\",external_web_access=false}} }}",
                        quoted(&base)
                    ),
                ] {
                    plan.args.extend(["-c".into(), value.into()]);
                }
                plan.set("LMA_API_KEY", key);
                // This custom provider has no OpenAI account dependency.
                plan.remove
                    .extend(["OPENAI_API_KEY".into(), "OPENAI_BASE_URL".into()]);
            }
            "claude" => {
                let mut values = BTreeMap::new();
                for (name, value) in [
                    ("ANTHROPIC_BASE_URL", root.as_str()),
                    ("ANTHROPIC_API_KEY", ""),
                    ("ANTHROPIC_MODEL", model.as_str()),
                    ("ANTHROPIC_DEFAULT_OPUS_MODEL", model.as_str()),
                    ("ANTHROPIC_DEFAULT_SONNET_MODEL", model.as_str()),
                    ("ANTHROPIC_DEFAULT_HAIKU_MODEL", model.as_str()),
                    ("ANTHROPIC_DEFAULT_FABLE_MODEL", model.as_str()),
                    ("CLAUDE_CODE_USE_BEDROCK", "0"),
                    ("CLAUDE_CODE_USE_VERTEX", "0"),
                    ("CLAUDE_CODE_USE_FOUNDRY", "0"),
                    ("CLAUDE_CODE_USE_ANTHROPIC_AWS", "0"),
                    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
                    ("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1"),
                    ("CLAUDE_CODE_DISABLE_THINKING", "1"),
                    ("DISABLE_PROMPT_CACHING", "1"),
                    ("ENABLE_TOOL_SEARCH", "false"),
                ] {
                    values.insert(name.to_string(), value.to_string());
                }
                values.insert(
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS".into(),
                    config.context.to_string(),
                );
                values.insert("CLAUDE_CODE_MAX_OUTPUT_TOKENS".into(), output.to_string());
                for (name, value) in &values {
                    plan.set(name, value);
                }
                // Credentials are environment-only, never --settings/argv data. Existing
                // approval/sandbox settings are retained; only inference settings are overlaid.
                plan.set("ANTHROPIC_AUTH_TOKEN", key);
                plan.remove.extend(
                    [
                        "CLAUDE_CODE_OAUTH_TOKEN",
                        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
                        "ANTHROPIC_CUSTOM_HEADERS",
                        "ANTHROPIC_BEDROCK_BASE_URL",
                        "ANTHROPIC_VERTEX_BASE_URL",
                        "ANTHROPIC_FOUNDRY_BASE_URL",
                        "ANTHROPIC_FOUNDRY_API_KEY",
                        "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
                        "ANTHROPIC_AWS_BASE_URL",
                        "ANTHROPIC_SMALL_FAST_MODEL",
                        "CLAUDE_CODE_SUBAGENT_MODEL",
                        "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
                    ]
                    .map(OsString::from),
                );
                plan.args.extend([
                    "--settings".into(),
                    json!({"env": values, "apiKeyHelper": ""})
                        .to_string()
                        .into(),
                    format!("--model={model}").into(),
                ]);
            }
            "opencode" => {
                // A fresh provider cannot inherit a saved per-model endpoint/package.
                // Alias the served ID so slashes/# never become selector syntax.
                let provider = format!(
                    "lma_launch_{}_{}",
                    std::process::id(),
                    NEXT_PROVIDER.fetch_add(1, Ordering::Relaxed)
                );
                let selected = format!("{provider}/selected");
                let mut profile = inline;
                let object = profile
                    .as_object_mut()
                    .ok_or("OPENCODE_CONFIG_CONTENT must be a JSON object")?;
                object.insert("model".into(), json!(&selected));
                object.insert("update".into(), json!("disable"));
                object.insert("small_model".into(), json!(&selected));
                let agents = object
                    .entry("agents")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .ok_or("OpenCode inline agents must be an object")?;
                let title = agents
                    .entry("title")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .ok_or("OpenCode inline title agent must be an object")?;
                title.insert("model".into(), json!(&selected));
                let providers = object
                    .entry("providers")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .ok_or("OpenCode inline providers must be an object")?;
                providers.insert(provider.clone(), json!({
                    "name": "lma (local)", "package": "@opencode/ai/providers/openai-compatible",
                    "settings": {"baseURL": base, "apiKey": "{env:LMA_API_KEY}", "transport": "http", "encoding": "none"},
                    "models": {"selected": {"modelID": model,
                        "capabilities": {"tools": true, "input": ["text"], "output": ["text"]},
                        "compatibility": {"maxTokensField": "max_tokens", "supportsPromptCacheKey": false},
                        "limit": {"context": config.context, "output": output}
                    }}
                }));
                // Legacy filters are explicitly supported by V2 normalization. Their
                // deny-by-default policy prevents implicit cloud/helper fallback.
                object.insert("enabled_providers".into(), json!([provider]));
                object.insert("disabled_providers".into(), json!([]));
                plan.args.push("--standalone".into());
                plan.set("OPENCODE_CONFIG_CONTENT", profile.to_string());
                plan.set("OPENCODE_DISABLE_MODELS_FETCH", "1");
                plan.set("OPENCODE_DISABLE_FREE_MODELS", "1");
                // A private server must not share a live service's SQLite database.
                plan.set("OPENCODE_DB", ":memory:");
                plan.set("LMA_API_KEY", key);
                plan.remove
                    .extend(["OPENCODE_API_KEY".into(), "OPENCODE_GO_API_KEY".into()]);
            }
            _ => return Err("unsupported external CLI target".into()),
        }
        if plan
            .args
            .iter()
            .chain(plan.env.values())
            .any(|s| s.as_encoded_bytes().contains(&0))
        {
            return Err("client launch configuration contains a NUL character".into());
        }
        Ok(plan)
    }

    fn set(&mut self, key: &str, value: impl AsRef<OsStr>) {
        self.env.insert(key.into(), value.as_ref().to_owned());
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command.args(&self.args).envs(&self.env);
        for key in &self.remove {
            command.env_remove(key);
        }
        command
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        command
    }
}

fn quoted(value: &str) -> String {
    serde_json::to_string(value)
        .expect("string serialization")
        .replace('\u{7f}', "\\u007f")
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

// All process syscalls, waits, and sleeps stay off the async executor. Dropping
// the stop sender requests cleanup even if the async supervisor is cancelled.
fn monitor_child(plan: ClientPlan, stop: mpsc::Receiver<()>) -> Result<(), String> {
    if !matches!(stop.try_recv(), Err(mpsc::TryRecvError::Empty)) {
        return Ok(());
    }
    let mut child = OwnedChild(
        plan.command()
            .spawn()
            .map_err(|error| format!("cannot start native {} CLI: {error}", plan.name))?,
    );
    loop {
        if let Some(status) = child
            .0
            .try_wait()
            .map_err(|error| format!("cannot poll {} CLI: {error}", plan.name))?
        {
            return exit_result(plan.name, status);
        }
        if !matches!(
            stop.recv_timeout(POLL),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            // Ctrl-C normally reached the inherited foreground terminal's child too.
            let deadline = Instant::now() + Duration::from_millis(500);
            while Instant::now() < deadline {
                if child
                    .0
                    .try_wait()
                    .map_err(|error| format!("cannot poll stopping client: {error}"))?
                    .is_some()
                {
                    return Ok(());
                }
                std::thread::sleep(POLL);
            }
            if let Err(error) = child.0.kill() {
                if child.0.try_wait().ok().flatten().is_none() {
                    return Err(format!("cannot stop owned {} CLI: {error}", plan.name));
                }
            }
            child
                .0
                .wait()
                .map_err(|error| format!("cannot reap owned {} CLI: {error}", plan.name))?;
            return Ok(());
        }
    }
}

fn exit_result(name: &str, status: ExitStatus) -> Result<(), String> {
    if status.success() {
        Ok(())
    } else {
        Err(format!("{name} CLI exited with {status}"))
    }
}

fn native_executable(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return false;
        }
    }
    let mut magic = [0u8; 4];
    if fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_err()
    {
        return false;
    }
    matches!(
        magic,
        [0x7f, b'E', b'L', b'F']
            | [0xfe, 0xed, 0xfa, 0xce]
            | [0xce, 0xfa, 0xed, 0xfe]
            | [0xfe, 0xed, 0xfa, 0xcf]
            | [0xcf, 0xfa, 0xed, 0xfe]
            | [0xca, 0xfe, 0xba, 0xbe]
            | [0xbe, 0xba, 0xfe, 0xca]
            | [0xca, 0xfe, 0xba, 0xbf]
            | [0xbf, 0xba, 0xfe, 0xca]
    ) || cfg!(windows) && magic[..2] == *b"MZ"
}

fn preflight_executable(name: &str) -> Result<PathBuf, String> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let mut candidates = Vec::new();
    for directory in std::env::split_paths(&path).take(256) {
        candidates.push(directory.join(name));
        #[cfg(windows)]
        candidates.push(directory.join(format!("{name}.exe")));
    }
    if name == "opencode" {
        if let Some(home) = &home {
            // Resolve an already-installed npm native payload without invoking npx,
            // package managers, postinstall scripts, or arbitrary shell wrappers.
            candidates.push(home.join(".opencode/bin/opencode"));
            if let Ok(entries) = fs::read_dir(home.join(".npm/_npx")) {
                let mut roots: Vec<_> = entries
                    .take(128)
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .collect();
                roots.sort();
                for root in roots {
                    candidates.push(root.join("node_modules/@opencode/cli/bin/opencode.exe"));
                }
            }
        }
    }
    let executable = candidates.into_iter().find(|path| native_executable(path)).ok_or_else(|| format!(
        "native {name} CLI not found; shell/JavaScript/npx installer shims are not executed. Install a native CLI separately or put an already-installed native executable on PATH; no model loaded"
    ))?;
    let executable = fs::canonicalize(executable)
        .map_err(|error| format!("cannot resolve native {name} executable: {error}"))?;
    let version = native_version(&executable)?;
    validate_version(name, &version)?;
    Ok(executable)
}

fn native_version(executable: &Path) -> Result<String, String> {
    let mut child = OwnedChild(
        Command::new(executable)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("cannot inspect native CLI version: {error}"))?,
    );
    let stdout = child
        .0
        .stdout
        .take()
        .ok_or("native CLI version stdout unavailable")?;
    let reader = std::thread::spawn(move || {
        let mut output = Vec::new();
        stdout.take(4097).read_to_end(&mut output).map(|_| output)
    });
    let deadline = Instant::now() + VERSION_TIMEOUT;
    loop {
        if let Some(status) = child
            .0
            .try_wait()
            .map_err(|error| format!("cannot inspect native CLI version: {error}"))?
        {
            if !status.success() {
                return Err("native CLI --version failed; no model loaded".into());
            }
            // A descendant retaining stdout must not make preflight wait forever.
            while !reader.is_finished() && Instant::now() < deadline {
                std::thread::sleep(POLL);
            }
            if !reader.is_finished() {
                return Err(
                    "native CLI version output did not close within 3 seconds; no model loaded"
                        .into(),
                );
            }
            let bytes = reader
                .join()
                .map_err(|_| "native CLI version reader failed")?
                .map_err(|error| error.to_string())?;
            if bytes.len() > 4096 {
                return Err("native CLI version output exceeds preflight limit".into());
            }
            return String::from_utf8(bytes).map_err(|_| "native CLI version is not UTF-8".into());
        }
        if Instant::now() >= deadline {
            return Err("native CLI --version timed out after 3 seconds; no model loaded".into());
        }
        std::thread::sleep(POLL);
    }
}

fn validate_version(name: &str, output: &str) -> Result<(), String> {
    let version = output
        .split_whitespace()
        .find_map(|word| {
            let word = word.trim_start_matches('v');
            let mut parts = word.split('.');
            let major = parts.next()?.parse::<u32>().ok()?;
            let minor = parts.next()?.parse::<u32>().ok()?;
            let patch = parts.next()?.parse::<u32>().ok()?;
            Some((major, minor, patch))
        })
        .ok_or_else(|| format!("cannot identify native {name} version; no model loaded"))?;
    let supported = match name {
        "opencode" => version.0 == 2 && version.1 == 2,
        "codex" => version.0 == 0 && version.1 == 163,
        "claude" => version.0 == 2 && version.1 == 1 && version.2 >= 285,
        _ => false,
    };
    if supported {
        Ok(())
    } else {
        Err(format!(
            "unsupported native {name} version {}.{}.{} for automatic local routing; verified profiles require OpenCode 2.2.x (--standalone), Codex 0.163.x, or Claude Code 2.1.285+ within 2.1.x. OpenCode V1/shared-service and desktop launchers are not substituted; no model loaded",
            version.0, version.1, version.2
        ))
    }
}

fn preflight_managed(name: &str) -> Result<(), String> {
    if name == "opencode" {
        return Ok(()); // Its managed and reversed provider-policy layers are checked separately.
    }
    let mut paths = match name {
        "codex" => vec![
            PathBuf::from("/etc/codex/managed_config.toml"),
            PathBuf::from("/etc/codex/requirements.toml"),
        ],
        "claude" => vec![
            PathBuf::from("/etc/claude-code/managed-settings.json"),
            PathBuf::from("/etc/claude-code/managed-settings.d"),
        ],
        _ => return Err("unsupported managed client profile".into()),
    };
    #[cfg(target_os = "macos")]
    {
        let domain = if name == "codex" {
            "com.openai.codex.plist"
        } else {
            "com.anthropic.claudecode.plist"
        };
        let root = PathBuf::from("/Library/Managed Preferences");
        paths.push(root.join(domain));
        if let Some(user) = std::env::var_os("USER") {
            paths.push(root.join(user).join(domain));
        }
        if name == "claude" {
            paths.push("/Library/Application Support/ClaudeCode/managed-settings.json".into());
            paths.push("/Library/Application Support/ClaudeCode/managed-settings.d".into());
        }
    }
    #[cfg(windows)]
    {
        if name == "codex" {
            if let Some(home) = std::env::var_os("CODEX_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("USERPROFILE").map(|p| PathBuf::from(p).join(".codex"))
                })
            {
                paths.push(home.join("managed_config.toml"));
            }
            if let Some(root) = std::env::var_os("ProgramData") {
                paths.push(PathBuf::from(root).join("OpenAI/Codex/requirements.toml"));
            }
        } else if let Some(root) = std::env::var_os("ProgramFiles") {
            paths.push(PathBuf::from(root).join("ClaudeCode/managed-settings.json"));
            paths.push(PathBuf::from(root).join("ClaudeCode/managed-settings.d"));
        }
    }
    for path in paths {
        match fs::metadata(path) {
            Ok(_) => {
                return Err(format!(
                    "{name} managed configuration is present and can override session routing; automatic localhost launch is refused without changing policy; no model loaded"
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(format!(
                    "cannot inspect {name} managed configuration; no model loaded"
                ));
            }
        }
    }
    Ok(())
}

fn inline_opencode() -> Result<Value, String> {
    match std::env::var("OPENCODE_CONFIG_CONTENT") {
        Ok(text) => {
            if text.len() as u64 > MAX_CONFIG {
                return Err("OpenCode inline config exceeds preflight limit".into());
            }
            serde_json::from_str(&text).map_err(|_| "existing OPENCODE_CONFIG_CONTENT must be valid JSON for a safe session overlay; no configuration was changed".into())
        }
        Err(std::env::VarError::NotPresent) => Ok(json!({})),
        Err(_) => Err("OPENCODE_CONFIG_CONTENT must be UTF-8".into()),
    }
}

fn preflight_opencode_config() -> Result<(), String> {
    // Managed routing outranks inline settings. Never claim a local connection
    // when a machine policy can replace the endpoint or re-enable cloud fallback.
    let mut managed = vec![PathBuf::from("/etc/opencode")];
    #[cfg(target_os = "macos")]
    {
        managed.push("/Library/Application Support/opencode".into());
        let root = PathBuf::from("/Library/Managed Preferences");
        managed.push(root.join("ai.opencode.managed.plist"));
        if let Some(user) = std::env::var_os("USER") {
            managed.push(root.join(user).join("ai.opencode.managed.plist"));
        }
    }
    #[cfg(windows)]
    if let Some(root) = std::env::var_os("ProgramData") {
        managed.push(PathBuf::from(root).join("opencode"));
    }
    for path in managed {
        match fs::metadata(path) {
            Ok(_) => return Err("OpenCode managed configuration is present; cannot guarantee per-launch localhost routing without changing policy; no model loaded".into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err("cannot inspect OpenCode managed configuration; no model loaded".into()),
        }
    }
    let mut paths = Vec::new();
    if let Some(root) = std::env::var_os("OPENCODE_CONFIG_DIR") {
        add_config_paths(&mut paths, &PathBuf::from(root));
    }
    if let Some(path) = std::env::var_os("OPENCODE_CONFIG") {
        paths.push(path.into());
    }
    let global = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")));
    if let Some(global) = global {
        add_config_paths(&mut paths, &global.join("opencode"));
    }
    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    for directory in cwd.ancestors() {
        add_config_paths(&mut paths, directory);
        add_config_paths(&mut paths, &directory.join(".opencode"));
    }
    for path in paths {
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => {
                return Err(
                    "cannot inspect an OpenCode configuration file; no model loaded".into(),
                );
            }
        };
        if metadata.len() > MAX_CONFIG {
            return Err("OpenCode configuration exceeds preflight limit; no model loaded".into());
        }
        let text = fs::read_to_string(&path)
            .map_err(|_| "cannot read an OpenCode configuration file; no model loaded")?;
        // JSONC needs a real client parser. Fail closed rather than misparse policy.
        let value: Value = serde_json::from_str(&text).map_err(|_| "OpenCode configuration uses JSONC or invalid JSON; automatic routing cannot safely verify policy. Use plain JSON or launch manually; no files changed and no model loaded")?;
        check_opencode_policy(&value)?;
    }
    check_opencode_policy(&inline_opencode()?)
}

fn add_config_paths(paths: &mut Vec<PathBuf>, directory: &Path) {
    paths.push(directory.join("opencode.json"));
    paths.push(directory.join("opencode.jsonc"));
}

fn check_opencode_policy(value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or("OpenCode configuration must be an object; no model loaded")?;
    if ["policies", "enabled_providers", "disabled_providers"]
        .iter()
        .any(|key| object.contains_key(*key))
        || object
            .get("experimental")
            .is_some_and(|v| v.get("policies").is_some())
    {
        return Err("OpenCode provider policy/filter is already configured; cannot safely override its authority for automatic local routing. Launch manually; no files changed and no model loaded".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ServeConfig {
        ServeConfig {
            model_id: Some("local/odd\"\\\n\t\u{7f}#model".into()),
            api_key: Some("secret-sentinel".into()),
            context: 8192,
            ..ServeConfig::default()
        }
    }

    fn plan(name: &'static str) -> ClientPlan {
        ClientPlan::new(
            name,
            PathBuf::from("/not-executed/native-client"),
            &config(),
            "127.0.0.1:12345".parse().unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn desktop_and_chat_preflight_never_resolve_models_or_executables() {
        for target in [
            LaunchTarget::OpencodeDesktop,
            LaunchTarget::CodexDesktop,
            LaunchTarget::ClaudeDesktop,
        ] {
            let error = run(
                ServeConfig {
                    model: "/missing/no-model.gguf".into(),
                    ..ServeConfig::default()
                },
                target,
            )
            .unwrap_err();
            assert!(error.contains("desktop local-provider setup required, not opened"));
            assert!(error.contains("CLI overrides do not configure desktop"));
        }
        assert!(cli_name(LaunchTarget::Chat).is_err());
        assert_eq!(cli_name(LaunchTarget::Opencode).unwrap(), "opencode");
        assert_eq!(cli_name(LaunchTarget::Codex).unwrap(), "codex");
        assert_eq!(cli_name(LaunchTarget::Claude).unwrap(), "claude");
    }

    #[test]
    fn only_loopback_and_valid_credentials_are_allowed() {
        for host in ["0.0.0.0", "::", "192.168.1.1", "example.com"] {
            assert!(
                launch_address(&ServeConfig {
                    host: host.into(),
                    ..config()
                })
                .is_err()
            );
        }
        for host in ["127.0.0.1", "localhost", "::1", "[::1]"] {
            assert!(
                launch_address(&ServeConfig {
                    host: host.into(),
                    port: 0,
                    ..config()
                })
                .unwrap()
                .ip()
                .is_loopback()
            );
        }
        for key in ["", "has space", "line\nfeed", "nul\0key"] {
            assert!(
                validate_key(&ServeConfig {
                    api_key: Some(key.into()),
                    ..config()
                })
                .is_err()
            );
        }
    }

    #[test]
    fn codex_profile_is_argument_data_and_secret_stays_in_child_environment() {
        let plan = plan("codex");
        let args: Vec<_> = plan.args.iter().map(|s| s.to_str().unwrap()).collect();
        assert_eq!(args[0], "--no-daemon");
        assert!(args[1..].chunks_exact(2).all(|p| p[0] == "-c"));
        let model = args[2].strip_prefix("model=").unwrap();
        assert_eq!(
            serde_json::from_str::<String>(model).unwrap(),
            config().model_id.unwrap()
        );
        let provider = args.last().unwrap();
        assert!(provider.contains("wire_api=\"responses\""));
        assert!(provider.contains("supports_websockets=false"));
        assert!(provider.contains("requires_openai_auth=false"));
        assert!(args.contains(&"features.enable_request_compression=false"));
        assert!(args.contains(&"web_search=\"disabled\""));
        assert_eq!(
            plan.env.get(OsStr::new("LMA_API_KEY")).unwrap(),
            "secret-sentinel"
        );
        assert!(args.iter().all(|s| !s.contains("secret-sentinel")));
    }

    #[test]
    fn claude_settings_pin_routing_without_approval_changes_or_secret_arguments() {
        let plan = plan("claude");
        assert_eq!(plan.args[0], "--settings");
        let settings: Value = serde_json::from_str(plan.args[1].to_str().unwrap()).unwrap();
        assert_eq!(
            settings["env"]["ANTHROPIC_BASE_URL"],
            "http://127.0.0.1:12345"
        );
        assert_eq!(settings["env"]["CLAUDE_CODE_USE_BEDROCK"], "0");
        assert_eq!(settings["env"]["ANTHROPIC_API_KEY"], "");
        assert!(settings.get("permissions").is_none());
        assert_eq!(
            plan.env.get(OsStr::new("ANTHROPIC_AUTH_TOKEN")).unwrap(),
            "secret-sentinel"
        );
        assert!(
            plan.args
                .iter()
                .all(|s| !s.to_string_lossy().contains("secret-sentinel"))
        );
        assert!(plan.remove.contains(&"CLAUDE_CODE_OAUTH_TOKEN".into()));
    }

    #[test]
    fn opencode_private_profile_pins_main_title_and_preserves_permissions() {
        let before = std::env::var_os("OPENCODE_CONFIG_CONTENT");
        let plan = ClientPlan::with_inline(
            "opencode",
            "/not-executed/native-client".into(),
            &config(),
            "[::1]:12345".parse().unwrap(),
            json!({
                "permissions": [{"action": "shell", "resource": "*", "effect": "ask"}],
                "agents": {"title": {"description": "keep this"}}
            }),
        )
        .unwrap();
        assert_eq!(plan.args, [OsString::from("--standalone")]);
        let profile: Value = serde_json::from_str(
            plan.env
                .get(OsStr::new("OPENCODE_CONFIG_CONTENT"))
                .unwrap()
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let selected = profile["model"].as_str().unwrap();
        let provider = selected.strip_suffix("/selected").unwrap();
        assert!(provider.starts_with("lma_launch_"));
        assert_eq!(profile["agents"]["title"]["model"], selected);
        assert_eq!(profile["agents"]["title"]["description"], "keep this");
        assert_eq!(profile["permissions"][0]["effect"], "ask");
        assert_eq!(profile["enabled_providers"], json!([provider]));
        assert_eq!(
            profile["providers"][provider]["models"]["selected"]["modelID"],
            config().model_id.unwrap()
        );
        assert_eq!(
            profile["providers"][provider]["settings"]["baseURL"],
            "http://[::1]:12345/v1"
        );
        assert!(!profile.to_string().contains("secret-sentinel"));
        assert_eq!(plan.env.get(OsStr::new("OPENCODE_DB")).unwrap(), ":memory:");
        assert_eq!(std::env::var_os("OPENCODE_CONFIG_CONTENT"), before);
    }

    #[test]
    fn native_version_contracts_fail_closed() {
        for (name, output) in [
            ("opencode", "opencode v2.2.2"),
            ("codex", "codex-cli 0.163.0"),
            ("claude", "2.1.312 (Claude Code)"),
        ] {
            assert!(validate_version(name, output).is_ok());
        }
        for (name, output) in [
            ("opencode", "1.2.3"),
            ("opencode", "3.0.0"),
            ("codex", "0.162.0"),
            ("claude", "2.1.284"),
            ("claude", "unknown"),
        ] {
            assert!(validate_version(name, output).is_err());
        }
    }

    #[test]
    fn socket_reservation_rejects_collisions_before_host_creation() {
        let listener = reserve_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
        let error = reserve_endpoint(listener.local_addr().unwrap()).unwrap_err();
        assert!(error.contains("cannot reserve local client endpoint"));
        assert!(error.contains("no model loaded"));
    }

    #[test]
    fn installer_shims_are_not_executable_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opencode");
        fs::write(&path, "#!/bin/sh\nnpx -y @opencode/cli \"$@\"\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert!(!native_executable(&path));
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "#!/bin/sh\nnpx -y @opencode/cli \"$@\"\n"
        );
    }

    #[test]
    fn existing_opencode_policies_require_manual_launch() {
        for value in [
            json!({"policies": []}),
            json!({"enabled_providers": ["openai"]}),
            json!({"experimental": {"policies": []}}),
        ] {
            assert!(check_opencode_policy(&value).is_err());
        }
        assert!(
            check_opencode_policy(
                &json!({"permissions": [{"action":"shell", "resource":"*", "effect":"ask"}]})
            )
            .is_ok()
        );
    }

    #[tokio::test]
    async fn supervisor_stops_fake_owned_child_and_drops_router_before_returning() {
        let (stop, requests) = mpsc::channel();
        let child = tokio::task::spawn_blocking(move || {
            requests
                .recv_timeout(Duration::from_secs(2))
                .map_err(|e| e.to_string())?;
            Ok(())
        });
        let (http_stop, http_done) = oneshot::channel();
        let (dropped, observed) = oneshot::channel();
        let http = tokio::spawn(async move {
            let _ = http_done.await;
            let _ = dropped.send(());
            Ok(())
        });
        supervise(child, stop, http, http_stop, std::future::ready(Ok(())))
            .await
            .unwrap();
        observed.await.unwrap();
    }

    #[test]
    fn cancelled_queued_launch_does_not_spawn_a_process() {
        let (stop, requests) = mpsc::channel();
        stop.send(()).unwrap();
        // This nonexistent executable would fail if monitor_child tried to spawn it.
        monitor_child(plan("codex"), requests).unwrap();
    }

    #[tokio::test]
    async fn http_failure_stops_fake_owned_child() {
        let (stop, requests) = mpsc::channel();
        let child = tokio::task::spawn_blocking(move || {
            requests
                .recv_timeout(Duration::from_secs(2))
                .map_err(|e| e.to_string())?;
            Ok(())
        });
        let (http_stop, _http_done) = oneshot::channel();
        let http = tokio::spawn(async { Err("fake listener failure".into()) });
        let error = supervise(child, stop, http, http_stop, std::future::pending())
            .await
            .unwrap_err();
        assert!(error.contains("fake listener failure"));
    }

    #[tokio::test]
    async fn child_failure_is_preserved_while_http_shuts_down() {
        let (stop, _requests) = mpsc::channel();
        let child = tokio::spawn(async { Err("fake client exited with status 7".into()) });
        let (http_stop, http_done) = oneshot::channel();
        let http = tokio::spawn(async move {
            let _ = http_done.await;
            Ok(())
        });
        let result = supervise(child, stop, http, http_stop, std::future::pending()).await;
        assert!(result.unwrap_err().contains("status 7"));
    }
}
