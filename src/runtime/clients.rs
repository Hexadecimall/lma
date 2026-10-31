//! Connection hints only: no process execution, environment mutation, or file writes.
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use crate::config::{ServeConfig, default_api_for_client};
use serde_json::{Value, json};

const REDACTED_KEY: &str = "REPLACE_WITH_YOUR_LMA_API_KEY";
const LOCAL_KEY: &str = "local-not-required";

/// Human-readable connection settings. Never reads client credentials or prints the server key.
/// Snippets are data, not shell commands; users decide whether and where to apply them.
pub fn setup(config: &ServeConfig, address: SocketAddr) -> String {
    let address = if address.ip().is_unspecified() {
        SocketAddr::new(
            if address.is_ipv6() {
                IpAddr::V6(Ipv6Addr::LOCALHOST)
            } else {
                IpAddr::V4(Ipv4Addr::LOCALHOST)
            },
            address.port(),
        )
    } else {
        address
    };
    let root = format!("http://{address}");
    let model = config.model_id.clone().unwrap_or_else(|| {
        config
            .model
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    let openai = matches!(config.api.as_str(), "all" | "openai");
    let anthropic = matches!(config.api.as_str(), "all" | "anthropic");
    let mut result = if openai {
        format!("Chat in your browser: {root}/\nOpenAI base URL: {root}/v1")
    } else if anthropic {
        format!("Anthropic base URL: {root} (POST /v1/messages)")
    } else {
        format!("Ollama chat URL: {root}/api/chat")
    };
    if config.api == "all" {
        result.push_str(&format!(
            "\nAnthropic base URL: {root} (POST /v1/messages)\nOllama chat URL: {root}/api/chat"
        ));
    }
    result.push_str(&format!("\nModel ID: {}", quoted(&model)));
    result.push_str(if config.api_key.is_some() {
        "\nAuthentication: set your configured API key locally; it is never printed here."
    } else {
        "\nAuthentication: not required by this server."
    });
    result.push_str("\nCompatibility: native text SSE; constrained/buffered tool args; client-executed tools; model-dependent, not full client parity.");

    let family = default_api_for_client(&config.client);
    if family == "all" {
        return result;
    }
    if config.api != "all" && config.api != family {
        result.push_str(&format!(
            "\nSelected client preset's API family is disabled by -a (NSS: --api); no usable client profile emitted. Enable {family}/all for {}.",
            config.client
        ));
        return result;
    }

    let key = if config.api_key.is_some() {
        REDACTED_KEY
    } else {
        LOCAL_KEY
    };
    let base = format!("{root}/v1");
    // Keep output conservative: reserve most context for client instructions/history/tools.
    let output_tokens = (config.context / 4).clamp(1, 256);
    result.push_str("\nConnection preset only: merge manually; nothing is written, exported, installed, or executed.\n");
    match config.client.as_str() {
        "claude-code" => {
            result
                .push_str("Claude Code gateway environment values (JSON; not a settings file):\n");
            json_block(
                &mut result,
                &json!({
                    "ANTHROPIC_BASE_URL": root,
                    "ANTHROPIC_AUTH_TOKEN": key,
                    "ANTHROPIC_MODEL": model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                    "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                    "CLAUDE_CODE_DISABLE_THINKING": "1",
                    "DISABLE_PROMPT_CACHING": "1",
                    "ENABLE_TOOL_SEARCH": "false",
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": config.context.to_string(),
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": output_tokens.to_string()
                }),
            );
            result.push_str("Set these in the environment of a separately launched Claude Code CLI, or merge under env in user ~/.claude/settings.json. AUTH_TOKEN sends Bearer auth; unset conflicting API keys/provider overrides. Family aliases all name the SAME local model, not Claude models. Nonessential traffic off also disables auto-updates; it is not a complete network sandbox.\n");
        }
        "codex" => {
            result.push_str(&format!(
                "Codex user-level ~/.codex/config.toml (not project .codex/config.toml):\n```toml\nmodel = {}\nmodel_provider = \"lma\"\nmodel_context_window = {}\nmodel_supports_reasoning_summaries = false\nweb_search = \"disabled\"\n\n[features]\nenable_request_compression = false\n\n[model_providers.lma]\nname = \"lma\"\nbase_url = {}\nwire_api = \"responses\"\nenv_key = \"LMA_API_KEY\"\nrequires_openai_auth = false\nsupports_websockets = false\n\n[model_providers.lma.capabilities]\nremote_compaction = \"unsupported\"\nexternal_web_access = false\n```\n",
                quoted(&model), config.context, quoted(&base)
            ));
            result.push_str("Set LMA_API_KEY locally in the separately launched Codex environment (JSON values):\n");
            json_block(&mut result, &json!({"LMA_API_KEY": key}));
            result.push_str("Use HTTP Responses/SSE, not WebSockets or compressed request bodies. Keep both transport switches off unless the server explicitly supports them. Project-local provider settings are ignored by current Codex. No hosted web search or remote compaction is supplied by lma.\n");
        }
        "zed" => {
            result.push_str("Zed user settings.json: merge language_models.openai_compatible (NOT ACP/external agents):\n");
            json_block(
                &mut result,
                &zed_profile(config, &base, &model, output_tokens),
            );
            result.push_str("Open agent: open settings, set the lma provider key in the system keychain, and select its model. Alternatively set LMA_API_KEY in the local Zed process environment (JSON values):\n");
            json_block(&mut result, &json!({"LMA_API_KEY": key}));
            result.push_str("Do not put keys in settings.json. This config powers Zed-owned AI features; it does not configure ACP, external agents, terminal threads, or edit prediction. Tools are enabled for capable local models; turn capabilities.tools off for chat-only models.\n");
        }
        "vscode" => {
            result.push_str(&format!(
                "VS Code with the Continue extension: merge ~/.continue/config.yaml (not VS Code settings.json):\n```yaml\nname: lma\nversion: 1.0.0\nschema: v1\nmodels:\n  - name: lma\n    provider: openai\n    model: {}\n    apiBase: {}\n    apiKey: {}\n    useResponsesApi: false\n    roles: [chat]\n    capabilities: [tool_use]\n    defaultCompletionOptions:\n      contextLength: {}\n      maxTokens: {}\n```\n",
                quoted(&model), quoted(&base), quoted(key), config.context, output_tokens
            ));
            result.push_str("vscode means Continue, not a universal VS Code/Copilot API. Install/select Continue yourself. Replace apiKey privately or use Continue secrets; never commit credentials. Remove tool_use for a chat-only model. No autocomplete, embedding, or reranking model is configured.\n");
        }
        "opencode" => {
            result
                .push_str("OpenCode opencode.json: merge this custom Chat Completions provider:\n");
            json_block(
                &mut result,
                &opencode_profile(config, &base, &model, key, output_tokens),
            );
            if config.api_key.is_some() {
                result.push_str("Set LMA_API_KEY privately in the separately launched OpenCode environment; options.apiKey reads it via {env:LMA_API_KEY}.\n");
            }
            result.push_str("Select lma's model with /models. Both model and small_model use the local provider. The npm field selects the SDK package; lma does not install it. No provider-hosted search, embeddings, or reasoning transport is implied.\n");
        }
        _ => unreachable!("named preset has a known API family"),
    }
    result.push_str("Protocol subset only; installed-client end-to-end compatibility is not verified. Tool execution and permissions belong to the client, never lma. Native text streaming differs from constrained/buffered tool arguments; tool reliability depends on the local model and chat template. Hosted tools, arbitrary provider fields, WebSockets, and cloud agent services are unsupported. See docs/clients.md for official references and limits.");
    result
}

// JSON quoting also handles YAML double-quoted scalars and TOML basic strings.
// TOML additionally forbids a literal DEL, which JSON would otherwise leave intact.
fn quoted(value: &str) -> String {
    serde_json::to_string(value)
        .expect("string serialization")
        .replace('\u{7f}', "\\u007f")
}

fn json_block(output: &mut String, value: &Value) {
    output.push_str("```json\n");
    output.push_str(
        &serde_json::to_string_pretty(value)
            .expect("JSON value serialization")
            .replace('\u{7f}', "\\u007f"),
    );
    output.push_str("\n```\n");
}

fn zed_profile(config: &ServeConfig, base: &str, model: &str, output_tokens: u32) -> Value {
    json!({"language_models": {"openai_compatible": {"lma": {
        "api_url": base,
        "available_models": [{
            "name": model, "display_name": "lma (local)",
            "max_tokens": config.context, "max_output_tokens": output_tokens,
            "capabilities": {
                "tools": true, "images": false, "parallel_tool_calls": false,
                "prompt_cache_key": false, "chat_completions": true,
                "interleaved_reasoning": false, "max_tokens_parameter": true
            }
        }]
    }}}})
}

fn opencode_profile(
    config: &ServeConfig,
    base: &str,
    model: &str,
    key: &str,
    output_tokens: u32,
) -> Value {
    json!({
        "$schema": "https://opencode.ai/config.json",
        "model": format!("lma/{model}"),
        "small_model": format!("lma/{model}"),
        "provider": {"lma": {
            "npm": "@ai-sdk/openai-compatible", "name": "lma (local)",
            "options": {"baseURL": base, "apiKey": if config.api_key.is_some() { "{env:LMA_API_KEY}" } else { key }},
            "models": {(model): {
                "name": "lma (local)",
                "limit": {"context": config.context, "output": output_tokens}
            }}
        }}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_blocks(output: &str) -> Vec<Value> {
        output
            .split("```json\n")
            .skip(1)
            .map(|part| serde_json::from_str(part.split("\n```").next().unwrap()).unwrap())
            .collect()
    }

    #[test]
    fn default_settings_keep_capabilities_concise_without_dumping_presets() {
        for client in ["default", "custom"] {
            let config = ServeConfig {
                client: client.into(),
                ..ServeConfig::default()
            };
            let output = setup(&config, "127.0.0.1:8080".parse().unwrap());
            let capabilities: Vec<_> = output
                .lines()
                .filter(|s| s.starts_with("Compatibility:"))
                .collect();
            assert_eq!(capabilities.len(), 1);
            assert!(capabilities[0].len() < 170);
            assert!(capabilities[0].contains("client-executed tools"));
            assert!(output.lines().count() <= 8);
            assert!(!output.contains("```"));
        }
    }

    #[test]
    fn every_profile_hides_keys_escapes_models_and_respects_api_gating() {
        let model = "model\"\n\\test\t\u{1b}\u{7f}";
        for client in ["claude-code", "codex", "zed", "vscode", "opencode"] {
            for authenticated in [false, true] {
                let mut config = ServeConfig {
                    client: client.into(),
                    api: default_api_for_client(client).into(),
                    model_id: Some(model.into()),
                    api_key: authenticated.then(|| "very-secret-value".into()),
                    ..ServeConfig::default()
                };
                let address = "[::]:1234".parse().unwrap();
                let output = setup(&config, address);
                assert!(output.contains("http://[::1]:1234"));
                assert!(!output.contains("very-secret-value"));
                assert!(!output.contains(model), "raw model leaked in {client}");
                assert!(output.contains("model\\\"\\n\\\\test"));
                assert!(output.contains("end-to-end compatibility is not verified"));
                if client != "opencode" {
                    assert!(output.contains(if authenticated {
                        REDACTED_KEY
                    } else {
                        LOCAL_KEY
                    }));
                }
                let blocks = json_blocks(&output);
                for block in &blocks {
                    assert!(block.is_object());
                }
                match client {
                    "claude-code" => assert_eq!(blocks[0]["ANTHROPIC_MODEL"], model),
                    "zed" => assert_eq!(
                        blocks[0]["language_models"]["openai_compatible"]["lma"]["available_models"]
                            [0]["name"],
                        model
                    ),
                    "opencode" => {
                        assert!(blocks[0]["provider"]["lma"]["models"].get(model).is_some())
                    }
                    "codex" | "vscode" => {
                        let prefix = if client == "codex" {
                            "model = "
                        } else {
                            "    model: "
                        };
                        let line = output
                            .lines()
                            .find_map(|line| line.strip_prefix(prefix))
                            .unwrap();
                        assert_eq!(serde_json::from_str::<String>(line).unwrap(), model);
                    }
                    _ => unreachable!(),
                }
                assert!(!output.contains('\u{1b}'));
                assert!(!output.contains('\u{7f}'));
                config.api = "all".into();
                assert!(!setup(&config, address).contains("no usable client profile"));
                for disabled in if client == "claude-code" {
                    ["openai", "ollama"]
                } else {
                    ["anthropic", "ollama"]
                } {
                    config.api = disabled.into();
                    let gated = setup(&config, address);
                    assert!(gated.contains("no usable client profile"));
                    assert!(!gated.contains("```"));
                    assert!(!gated.contains("very-secret-value"));
                }
            }
        }
        assert_eq!(
            serde_json::from_str::<String>(&quoted(model)).unwrap(),
            model
        );
        assert!(!quoted(model).contains('\u{7f}'));
    }

    #[test]
    fn zed_uses_current_openai_compatible_provider_not_acp() {
        let config = ServeConfig {
            client: "zed".into(),
            context: 8192,
            ..ServeConfig::default()
        };
        let output = setup(&config, "0.0.0.0:8081".parse().unwrap());
        let blocks = json_blocks(&output);
        let provider = &blocks[0]["language_models"]["openai_compatible"]["lma"];
        assert_eq!(provider["api_url"], "http://127.0.0.1:8081/v1");
        assert_eq!(provider["available_models"][0]["max_tokens"], 8192);
        assert_eq!(
            provider["available_models"][0]["capabilities"]["max_tokens_parameter"],
            true
        );
        assert!(blocks[0].get("agent_servers").is_none());
        assert_eq!(blocks[1]["LMA_API_KEY"], LOCAL_KEY);
        assert!(!blocks[0].to_string().contains("api_key"));
    }

    #[test]
    fn opencode_uses_chat_sdk_env_auth_and_local_small_model() {
        let config = ServeConfig {
            client: "opencode".into(),
            model_id: Some("org/model\"".into()),
            api_key: Some("secret".into()),
            ..ServeConfig::default()
        };
        let output = setup(&config, "127.0.0.1:1234".parse().unwrap());
        let value = &json_blocks(&output)[0];
        assert_eq!(value["provider"]["lma"]["npm"], "@ai-sdk/openai-compatible");
        assert_eq!(
            value["provider"]["lma"]["options"]["apiKey"],
            "{env:LMA_API_KEY}"
        );
        assert_eq!(value["model"], "lma/org/model\"");
        assert_eq!(value["model"], value["small_model"]);
        assert!(
            value["provider"]["lma"]["models"]
                .get("org/model\"")
                .is_some()
        );
    }

    #[test]
    fn codex_uses_user_level_responses_without_websockets_or_compression() {
        let config = ServeConfig {
            client: "codex".into(),
            ..ServeConfig::default()
        };
        let output = setup(&config, "127.0.0.1:8080".parse().unwrap());
        for expected in [
            "user-level ~/.codex/config.toml",
            "wire_api = \"responses\"",
            "env_key = \"LMA_API_KEY\"",
            "requires_openai_auth = false",
            "supports_websockets = false",
            "enable_request_compression = false",
            "web_search = \"disabled\"",
        ] {
            assert!(output.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn claude_gateway_pins_family_aliases_without_leaking_credentials() {
        let config = ServeConfig {
            client: "claude-code".into(),
            model_id: Some("local.gguf".into()),
            ..ServeConfig::default()
        };
        let output = setup(&config, "127.0.0.1:8080".parse().unwrap());
        let value = &json_blocks(&output)[0];
        for name in [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_FABLE_MODEL",
        ] {
            assert_eq!(value[name], "local.gguf");
        }
        assert_eq!(value["ANTHROPIC_AUTH_TOKEN"], LOCAL_KEY);
        assert_eq!(value["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1");
        assert_eq!(value["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"], "1");
    }

    #[test]
    fn vscode_is_continue_yaml_with_bounded_chat_not_universal_vscode() {
        let config = ServeConfig {
            client: "vscode".into(),
            context: 2,
            ..ServeConfig::default()
        };
        let output = setup(&config, "127.0.0.1:8080".parse().unwrap());
        for expected in [
            "Continue extension",
            "~/.continue/config.yaml",
            "schema: v1",
            "provider: openai",
            "useResponsesApi: false",
            "roles: [chat]",
            "capabilities: [tool_use]",
            "contextLength: 2",
            "maxTokens: 1",
            "not a universal VS Code/Copilot API",
        ] {
            assert!(output.contains(expected), "missing {expected}");
        }
    }
}
