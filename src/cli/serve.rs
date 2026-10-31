use std::path::PathBuf;

use super::{
    Action, Command, Syntax,
    args::{Args, Flag, size, threads},
    help, unsupported,
};
use crate::config::{
    ALL_GPU_LAYERS, CLIENT_PRESETS, Job, LaunchConfig, LaunchTarget, Plan, ServeConfig,
    default_api_for_client,
};

pub(super) fn parse(mut args: Args, command: Command) -> Result<Action, String> {
    let allow_missing_model = matches!(command, Command::Launch | Command::Chat);

    let mut interactive = false;
    let mut target = LaunchTarget::default();
    let mut explicit_api = false;
    let mut syntax = Syntax::Friendly;
    let mut config = ServeConfig::default();
    let mut model = None;
    let mut micro_batch = None;
    let mut dry_run = false;
    while let Some(argument) = args.next() {
        if argument == "--" {
            syntax = Syntax::Nss;
            continue;
        }
        let Some(flag) = Flag::parse(&argument) else {
            if syntax == Syntax::Nss {
                return Err("after --, select a model with --model NAME or --path FILE instead of a positional argument".into());
            }
            if model.replace(PathBuf::from(argument)).is_some() {
                return Err("model specified twice".into());
            }
            continue;
        };
        let friendly = syntax == Syntax::Friendly;
        let name = match flag.name.as_str() {
            "-h" if friendly => "--help",
            "-m" if friendly => "--model",
            "-o" if friendly => "--ollama",
            "-od" if friendly => "--ollama-dir",
            "-c" | "--context" if friendly => "--ctx-size",
            "-t" if friendly => "--threads",
            "--batch" if friendly => "--batch-size",
            "--micro-batch" if friendly => "--ubatch-size",
            "--flash" if friendly => "--flash-attn",
            other => other,
        };
        match name {
            "--help" | "--help-all" => {
                flag.switch()?;
                return Ok(Action::Help(match command {
                    Command::Launch => help::LAUNCH,
                    Command::Chat => help::CHAT,
                    _ => help::SERVE,
                }));
            }
            "--dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            "--tokens" => {
                if !allow_missing_model {
                    return Err("reply tokens are configured in terminal chat (`lma c -ts N|off`); HTTP clients supply max_tokens per request".into());
                }
                config.chat_tokens = Some(crate::config::chat_token_limit(&args.text(&flag)?)?);
            }
            "--target" => {
                if !matches!(command, Command::Launch) {
                    return Err("--target is only supported by `lma lh` / `lma nss launch`; use --client for print-only hosting presets".into());
                }
                target = LaunchTarget::parse(&args.text(&flag)?)?;
            }
            "--mode" => {
                return Err(
                    "--mode has been removed; use `lma c` for chat or `lma s` for HTTP hosting"
                        .into(),
                );
            }
            "--interactive" if allow_missing_model => {
                flag.switch()?;
                interactive = true;
            }
            "--path" => {
                let path = PathBuf::from(args.value(&flag)?);
                // Keep explicit file intent visible to the shared resolver, even
                // when a bare filename collides with an installed model name.
                let path = if path.is_relative() {
                    PathBuf::from(".").join(path)
                } else {
                    path
                };
                if model.replace(path).is_some() {
                    return Err("model specified twice".into());
                }
            }
            "--model" => {
                if model.replace(PathBuf::from(args.value(&flag)?)).is_some() {
                    return Err("model specified twice".into());
                }
            }
            "--ollama" => {
                let name = args.text(&flag)?;
                if model
                    .replace(PathBuf::from(format!("ollama:{name}")))
                    .is_some()
                {
                    return Err("model specified twice".into());
                }
            }
            "--ollama-dir" => config.ollama_dir = Some(PathBuf::from(args.value(&flag)?)),
            "--projector" | "--mmproj" => {
                config.projector = Some(PathBuf::from(args.value(&flag)?))
            }
            "--chat-template" => {
                config.chat_template = args.text(&flag)?;
                if !crate::config::CHAT_TEMPLATES.contains(&config.chat_template.as_str()) {
                    return Err(format!(
                        "--chat-template must be one of: {}",
                        crate::config::CHAT_TEMPLATES.join(", ")
                    ));
                }
                mark_explicit_load(&mut config, "template");
            }
            "--prompt-cache" => {
                flag.switch()?;
                config.prompt_cache = true;
            }
            "--no-prompt-cache" => {
                flag.switch()?;
                config.prompt_cache = false;
            }
            "--kv-cache" => {
                config.kv_cache = args.text(&flag)?;
                if !matches!(config.kv_cache.as_str(), "f16" | "q8_0" | "q4_0") {
                    return Err("--kv-cache must be f16, q8_0, or q4_0".into());
                }
            }
            "--memory-fallback" => {
                config.memory_fallback = args.text(&flag)?;
                if !matches!(config.memory_fallback.as_str(), "error" | "mmap") {
                    return Err("--memory-fallback must be error or mmap".into());
                }
            }
            "--expert-cache-mib" => {
                let value = args.text(&flag)?;
                let mib = value
                    .parse::<u32>()
                    .ok()
                    .filter(|mib| (1..=1_048_576).contains(mib))
                    .ok_or(
                        "--expert-cache-mib must be a positive integer between 1 and 1048576 MiB",
                    )?;
                usize::try_from(mib)
                    .ok()
                    .and_then(|mib| mib.checked_mul(1024 * 1024))
                    .ok_or("--expert-cache-mib byte budget is too large for this platform")?;
                config.expert_cache_mib = mib;
            }
            "--gpu" if friendly => {
                let value = args.text(&flag)?;
                config.gpu_layers = match value.as_str() {
                    "all" => ALL_GPU_LAYERS,
                    "cpu" => 0,
                    _ => integer(&value, "--gpu", true)?,
                };
                mark_explicit_load(&mut config, "num_gpu");
            }
            "--n-gpu-layers" => {
                config.gpu_layers = integer(&args.text(&flag)?, name, true)?;
                mark_explicit_load(&mut config, "num_gpu");
            }
            "--ctx-size" => {
                config.context = integer(&args.text(&flag)?, name, false)?;
                mark_explicit_load(&mut config, "num_ctx");
            }
            "--threads" => {
                config.threads = threads(&args.text(&flag)?)?
                    .parse()
                    .map_err(|_| "thread count is too large")?;
                mark_explicit_load(&mut config, "num_thread");
            }
            "--batch-size" => {
                config.batch = integer(&args.text(&flag)?, name, false)?;
                mark_explicit_load(&mut config, "num_batch");
            }
            "--ubatch-size" => {
                micro_batch = Some(integer(&args.text(&flag)?, name, false)?);
                mark_explicit_load(&mut config, "u_batch");
            }
            "--flash-attn" => {
                config.flash = args.text(&flag)?;
                if !matches!(config.flash.as_str(), "auto" | "on" | "off") {
                    return Err("flash attention must be auto, on, or off".into());
                }
            }
            "--api" => {
                config.api = args.text(&flag)?;
                if !matches!(
                    config.api.as_str(),
                    "all" | "openai" | "ollama" | "anthropic"
                ) {
                    return Err("--api must be all, openai, ollama, or anthropic".into());
                }
                explicit_api = true;
            }
            "--client" => {
                config.client = args.text(&flag)?;
                if !CLIENT_PRESETS.contains(&config.client.as_str()) {
                    return Err(format!(
                        "--client must be one of: {}",
                        CLIENT_PRESETS.join(", ")
                    ));
                }
            }
            "--host" => config.host = args.text(&flag)?,
            "--port" => {
                let port = integer(&args.text(&flag)?, name, false)?;
                config.port =
                    u16::try_from(port).map_err(|_| "port must be between 1 and 65535")?;
            }
            "--api-key" => {
                let key = args.text(&flag)?;
                if key.chars().any(char::is_whitespace) {
                    return Err("API key cannot contain whitespace".into());
                }
                config.api_key = Some(key);
            }
            _ => return Err(unsupported(&flag.name)),
        }
    }
    config.model = match model.filter(|path| !path.as_os_str().is_empty()) {
        Some(model) => model,
        None if allow_missing_model => PathBuf::new(),
        None => return Err("serve needs an installed model NAME or file path; try `lma s llama3.2`, `lma s -p model.gguf`, or `lma nss serve --path model.gguf`".into()),
    };
    if !explicit_api {
        config.api = default_api_for_client(&config.client).into();
    }
    if config.kv_cache != "f16" && config.flash != "on" {
        return Err("quantized KV cache requires --flash on (or --flash-attn on)".into());
    }
    if config.context < 2 {
        return Err("context must be at least 2 tokens".into());
    }
    if config.expert_cache_mib > 0 {
        if config.projector.is_some() {
            return Err(
                "--expert-cache-mib does not support a vision projector (--projector/--mmproj)"
                    .into(),
            );
        }
    }
    if config
        .chat_tokens
        .is_some_and(|tokens| tokens >= config.context)
    {
        return Err("chat reply tokens must be smaller than context (-c); use -ts off for remaining-context generation".into());
    }
    config.micro_batch = micro_batch.unwrap_or(config.micro_batch.min(config.batch));
    if config.micro_batch > config.batch {
        return Err("micro-batch cannot exceed batch size".into());
    }
    let job = if allow_missing_model {
        interactive |= config.model.as_os_str().is_empty();
        Job::Launch(LaunchConfig {
            serve: config,
            mode: if matches!(command, Command::Launch) {
                "launch"
            } else {
                "chat"
            }
            .into(),
            target,
            interactive,
        })
    } else {
        Job::Serve(config)
    };
    Ok(Action::Run(Box::new(Plan { job, dry_run })))
}

fn mark_explicit_load(config: &mut ServeConfig, key: &str) {
    if !config.explicit_load.iter().any(|value| value == key) {
        config.explicit_load.push(key.into());
    }
}

fn integer(value: &str, name: &str, allow_zero: bool) -> Result<u32, String> {
    size(value, name, allow_zero)?
        .parse()
        .map_err(|_| format!("{name} is too large"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    #[test]
    fn definition_load_overrides_track_canonical_keys_for_all_cli_aliases() {
        for prefix in [
            vec!["nss", "serve", "tiny"],
            vec!["nss", "server", "tiny"],
            vec!["nss", "chat", "tiny"],
            vec!["nss", "launch", "tiny"],
            vec!["nss", "lh", "tiny"],
        ] {
            let base = match plan(&prefix).job {
                Job::Serve(config) => config,
                Job::Launch(config) => config.serve,
                _ => panic!(),
            };
            assert!(base.definition.is_none());
            assert!(base.explicit_load.is_empty());
            for (flag, value, key) in [
                ("-c", "4096", "num_ctx"),
                ("--context", "4096", "num_ctx"),
                ("--ctx-size", "4096", "num_ctx"),
                ("--gpu", "cpu", "num_gpu"),
                ("--n-gpu-layers", "0", "num_gpu"),
                ("-t", "auto", "num_thread"),
                ("--threads", "2", "num_thread"),
                ("--batch", "32", "num_batch"),
                ("--batch-size", "512", "num_batch"),
                ("--micro-batch", "128", "u_batch"),
                ("--ubatch-size", "128", "u_batch"),
                ("--chat-template", "auto", "template"),
            ] {
                let mut argv = prefix.clone();
                argv.extend([flag, value, "--dry-run"]);
                let config = match plan(&argv).job {
                    Job::Serve(config) => config,
                    Job::Launch(config) => config.serve,
                    _ => panic!(),
                };
                assert_eq!(config.explicit_load, vec![key], "{argv:?}");
                assert!(config.definition.is_none());
                assert!(
                    serde_json::to_value(config)
                        .unwrap()
                        .get("explicit_load")
                        .is_none()
                );
            }
        }
    }

    #[test]
    fn canonical_hosting_flags_preserve_load_overrides_paths_and_expert_cache() {
        for command in ["s", "c", "lh"] {
            for (flag, value, key) in [
                ("-c", "4096", "num_ctx"),
                ("-g", "cpu", "num_gpu"),
                ("-t", "2", "num_thread"),
                ("-b", "512", "num_batch"),
                ("-mh", "128", "u_batch"),
                ("-ce", "auto", "template"),
            ] {
                let config = match plan(&[command, "tiny", flag, value, "-d"]).job {
                    Job::Serve(config) => config,
                    Job::Launch(config) => config.serve,
                    _ => panic!(),
                };
                assert_eq!(config.explicit_load, vec![key]);
                assert!(config.definition.is_none());
                assert!(
                    serde_json::to_value(config)
                        .unwrap()
                        .get("explicit_load")
                        .is_none()
                );
            }
            for (flags, expected) in [
                (vec!["-p", "tiny"], PathBuf::from(".").join("tiny")),
                (vec!["-p=tiny.gguf"], PathBuf::from(".").join("tiny.gguf")),
                (vec!["-m", "tiny"], PathBuf::from("tiny")),
                (vec!["tiny"], PathBuf::from("tiny")),
            ] {
                let mut argv = vec![command];
                argv.extend(flags);
                argv.extend(["-e=64", "-d"]);
                let parsed = plan(&argv);
                assert!(parsed.dry_run);
                let config = match parsed.job {
                    Job::Serve(config) => config,
                    Job::Launch(config) => config.serve,
                    _ => panic!(),
                };
                assert_eq!(config.model, expected);
                assert_eq!(config.expert_cache_mib, 64);
                assert_eq!(config.micro_batch, 128);
                assert_eq!(config.gpu_layers, crate::config::default_gpu_layers());
                assert!(config.explicit_load.is_empty());
                assert!(config.definition.is_none());
            }
        }
    }

    #[test]
    fn definition_load_override_keys_are_deduplicated_without_marking_implicit_micro_batch() {
        let Job::Serve(config) = plan(&[
            "nss",
            "serve",
            "tiny",
            "--context=4096",
            "--ctx-size=8192",
            "--gpu=cpu",
            "--n-gpu-layers=0",
            "--threads=2",
            "--batch=32",
            "--chat-template=auto",
            "--chat-template=plain",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(
            config.explicit_load,
            vec!["num_ctx", "num_gpu", "num_thread", "num_batch", "template"]
        );
        assert_eq!(config.context, 8192);
        assert_eq!(config.micro_batch, 32);
        assert_eq!(config.chat_template, "plain");
        assert!(!config.explicit_load.iter().any(|key| key == "u_batch"));

        let Job::Serve(config) = plan(&[
            "nss",
            "serve",
            "tiny",
            "--",
            "--ubatch-size=16",
            "--ctx-size=8192",
            "--chat-template=auto",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.explicit_load, vec!["u_batch", "num_ctx", "template"]);
    }

    #[test]
    fn expert_cache_is_a_host_knob_without_modelfile_override_keys() {
        for prefix in [
            vec!["nss", "serve"],
            vec!["nss", "server"],
            vec!["nss", "chat"],
            vec!["nss", "launch"],
            vec!["nss", "lh"],
        ] {
            let mut argv = prefix;
            argv.extend([
                "--model",
                "missing.gguf",
                "--expert-cache-mib=64",
                "--dry-run",
            ]);
            let config = match plan(&argv).job {
                Job::Serve(config) => config,
                Job::Launch(config) => config.serve,
                _ => panic!(),
            };
            assert_eq!(config.expert_cache_mib, 64);
            assert_eq!(config.micro_batch, 128);
            assert_eq!(config.gpu_layers, crate::config::default_gpu_layers());
            assert!(config.explicit_load.is_empty());
            assert!(config.definition.is_none());
        }
    }

    #[test]
    fn gpu_defaults_and_explicit_cpu_are_independent_of_fallback() {
        for fallback in ["error", "mmap"] {
            for (flags, expected) in [
                (vec![], crate::config::default_gpu_layers()),
                (vec!["--gpu", "cpu"], 0),
                (vec!["--gpu", "0"], 0),
                (vec!["--n-gpu-layers", "0"], 0),
                (vec!["--gpu", "1"], 1),
                (vec!["--gpu", "all"], ALL_GPU_LAYERS),
            ] {
                let mut argv = vec!["nss", "serve", "model.gguf", "--memory-fallback", fallback];
                argv.extend(flags);
                let Job::Serve(config) = plan(&argv).job else {
                    panic!()
                };
                assert_eq!(config.gpu_layers, expected);
                assert_eq!(config.memory_fallback, fallback);
            }
        }
    }

    #[test]
    fn shortcuts_build_typed_configuration() {
        let plan = plan(&[
            "s",
            "model.gguf",
            "-g",
            "all",
            "-c",
            "8k",
            "-f=on",
            "-t",
            "4",
            "-b",
            "64",
        ]);
        let Job::Serve(config) = plan.job else {
            panic!()
        };
        assert_eq!(config.context, 8192);
        assert_eq!(config.gpu_layers, 999);
        assert_eq!(config.threads, 4);
        assert_eq!(config.flash, "on");
        assert_eq!(config.micro_batch, 64);
    }

    #[test]
    fn ollama_option_aliases_preserve_model_tags_including_nss() {
        for argv in [
            vec!["s", "-o", "llama3.2:latest"],
            vec!["s", "-o=llama3.2:latest"],
            vec!["nss", "serve", "--ollama", "llama3.2:latest"],
            vec!["nss", "serve", "-o", "llama3.2:latest"],
            vec!["nss", "server", "-o=llama3.2:latest"],
        ] {
            let Job::Serve(config) = plan(&argv).job else {
                panic!()
            };
            assert_eq!(config.model, PathBuf::from("ollama:llama3.2:latest"));
        }
    }

    #[test]
    fn ollama_directory_alias_supports_inline_values_and_nss() {
        for argv in [
            vec!["s", "-o", "llama3.2", "-or", "my models"],
            vec!["s", "-o", "llama3.2", "-or=my models"],
            vec![
                "nss",
                "serve",
                "-o",
                "llama3.2",
                "--ollama-dir",
                "my models",
            ],
            vec!["nss", "serve", "-o", "llama3.2", "-od", "my models"],
            vec!["nss", "server", "-o", "llama3.2", "-od=my models"],
        ] {
            let Job::Serve(config) = plan(&argv).job else {
                panic!()
            };
            assert_eq!(config.ollama_dir, Some(PathBuf::from("my models")));
            assert_eq!(config.model, PathBuf::from("ollama:llama3.2"));
        }
    }

    #[test]
    fn nss_long_options_work_after_separator() {
        let plan = plan(&[
            "nss",
            "serve",
            "model.gguf",
            "--",
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "0",
        ]);
        let Job::Serve(config) = plan.job else {
            panic!()
        };
        assert_eq!(config.context, 8192);
        assert_eq!(config.gpu_layers, 0);
    }

    #[test]
    fn launch_and_chat_share_options_but_keep_distinct_modes() {
        for (prefix, flags) in [
            (vec!["lh"], vec!["-d", "-ct", "claude-code"]),
            (vec!["c"], vec!["-d", "-ct", "claude-code"]),
            (
                vec!["nss", "chat"],
                vec!["--dry-run", "--client", "claude-code"],
            ),
            (
                vec!["nss", "launch"],
                vec!["--dry-run", "--client", "claude-code"],
            ),
            (
                vec!["nss", "lh"],
                vec!["--dry-run", "--client", "claude-code"],
            ),
        ] {
            let expected_mode = if matches!(prefix.last(), Some(&"chat" | &"c")) {
                "chat"
            } else {
                "launch"
            };
            let mut argv = prefix;
            argv.extend(flags);
            let parsed = plan(&argv);
            assert!(parsed.dry_run);
            let Job::Launch(config) = parsed.job else {
                panic!()
            };
            assert!(config.serve.model.as_os_str().is_empty());
            assert!(config.interactive);
            assert_eq!(config.mode, expected_mode);
            assert_eq!(config.target, LaunchTarget::Chat);
            assert_eq!(config.serve.api, "anthropic");
        }
        let Job::Launch(config) = plan(&["lh", "m.gguf", "-i"]).job else {
            panic!()
        };
        assert_eq!(config.mode, "launch");
        assert_eq!(config.target, LaunchTarget::Chat);
        assert!(config.interactive);
        for argv in [
            vec!["nss", "chat", "--mode=host"],
            vec!["nss", "chat", "--mode=chat"],
            vec!["nss", "launch", "--mode=chat"],
            vec!["nss", "launch", "--mode=host"],
            vec!["nss", "launch", "--mode=agent"],
            vec!["nss", "serve", "m", "--mode=chat"],
            vec!["nss", "serve", "m", "--interactive"],
            vec!["nss", "launch", "m", "--", "-m", "n"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn launch_targets_are_typed_and_independent_of_client_presets() {
        for (prefix, target_flag, dry_run) in [
            (vec!["lh"], "-tt", "-d"),
            (vec!["nss", "launch"], "--target", "--dry-run"),
            (vec!["nss", "lh"], "--target", "--dry-run"),
        ] {
            for target in LaunchTarget::ALL {
                let inline = format!("{target_flag}={}", target.as_str());
                let mut argv = prefix.clone();
                argv.extend(["my-model:tag", &inline, dry_run]);
                let parsed = plan(&argv);
                assert!(parsed.dry_run);
                let Job::Launch(config) = parsed.job else {
                    panic!()
                };
                assert_eq!(config.target, target);
                assert_eq!(config.mode, "launch");
                assert_eq!(config.serve.model, PathBuf::from("my-model:tag"));
                assert!(!config.interactive);
                assert_eq!(config.serve.client, "default");
                assert_eq!(config.serve.api, "all");
            }
        }
        let Job::Launch(config) = plan(&[
            "lh",
            "-tt",
            "codex",
            "-ct",
            "claude-code",
            "-a",
            "ollama",
            "-d",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.target, LaunchTarget::Codex);
        assert_eq!(config.serve.client, "claude-code");
        assert_eq!(config.serve.api, "ollama");
        assert!(config.interactive);

        let Job::Launch(config) = plan(&[
            "nss",
            "launch",
            "--",
            "--target",
            "claude-desktop",
            "--model",
            "tiny",
            "--dry-run",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.target, LaunchTarget::ClaudeDesktop);
        assert_eq!(config.serve.model, PathBuf::from("tiny"));
    }

    #[test]
    fn target_flags_reject_invalid_values_and_non_launch_commands() {
        for argv in [
            vec!["lh", "-tt"],
            vec!["lh", "-tt="],
            vec!["lh", "-tt", "desktop"],
            vec!["nss", "launch", "--target", "CODEX"],
            vec!["c", "-tt", "chat"],
            vec!["nss", "chat", "--target=codex"],
            vec!["nss", "chat", "--target=chat"],
            vec!["s", "m", "-tt=codex"],
            vec!["nss", "serve", "m", "--target=chat"],
            vec!["lh", "one", "two", "-tt=codex"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn explicit_paths_and_installed_names_share_selection_safety() {
        for prefix in [
            vec!["nss", "serve"],
            vec!["nss", "server"],
            vec!["nss", "chat"],
            vec!["nss", "launch"],
            vec!["nss", "lh"],
        ] {
            for (flags, expected) in [
                (vec!["--path", "tiny"], PathBuf::from(".").join("tiny")),
                (
                    vec!["--path=tiny.gguf"],
                    PathBuf::from(".").join("tiny.gguf"),
                ),
                (vec!["--model", "tiny"], PathBuf::from("tiny")),
                (vec!["tiny"], PathBuf::from("tiny")),
            ] {
                let mut argv = prefix.clone();
                argv.extend(flags);
                argv.push("--dry-run");
                let config = match plan(&argv).job {
                    Job::Serve(config) => config,
                    Job::Launch(config) => config.serve,
                    _ => panic!(),
                };
                assert_eq!(config.model.as_os_str(), expected.as_os_str());
            }
        }
        for argv in [
            vec!["s", "-p="],
            vec!["s", "-p"],
            vec!["c", "-p", "a", "-m", "b"],
            vec!["s", "a", "-p", "b"],
            vec!["lh", "-p", "a", "-o", "b"],
            vec!["nss", "serve", "--path="],
            vec!["nss", "serve", "--path"],
            vec!["nss", "chat", "--path", "a", "--model", "b"],
            vec!["nss", "serve", "a", "--path", "b"],
            vec!["nss", "launch", "--path", "a", "-o", "b"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn api_defaults_follow_final_client_without_overriding_explicit_api() {
        for client in ["codex", "zed", "vscode", "opencode"] {
            let inline = format!("--client={client}");
            let Job::Serve(config) = plan(&["nss", "serve", "m", &inline]).job else {
                panic!()
            };
            assert_eq!(config.api, "openai");
            for flags in [
                ["--api=anthropic", inline.as_str()],
                [inline.as_str(), "--api=anthropic"],
            ] {
                let Job::Serve(config) = plan(&["nss", "serve", "m", flags[0], flags[1]]).job
                else {
                    panic!()
                };
                assert_eq!(config.api, "anthropic");
            }
        }
        for argv in [
            vec!["nss", "serve", "m", "--api=ollama", "--client=claude-code"],
            vec!["nss", "serve", "m", "--client=codex", "--api=ollama"],
        ] {
            let Job::Serve(config) = plan(&argv).job else {
                panic!()
            };
            assert_eq!(config.api, "ollama");
        }
        let Job::Serve(config) = plan(&[
            "nss",
            "serve",
            "m",
            "--client=claude-code",
            "--client=custom",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.api, "all");
        for (flag, value) in [("--api", "bad"), ("--client", "bad")] {
            assert!(
                crate::cli::parse(["nss", "serve", "m", flag, value].map(OsString::from)).is_err()
            );
        }
    }

    #[test]
    fn invalid_or_unimplemented_options_are_errors() {
        for argv in [
            vec!["nss", "serve"],
            vec!["nss", "serve", "m", "--context", "0"],
            vec!["nss", "serve", "m", "--port", "65536"],
            vec!["nss", "serve", "m", "--threads", "0"],
            vec!["nss", "serve", "m", "--micro-batch", "1024"],
            vec!["nss", "serve", "m", "--flash", "yes"],
            vec!["nss", "serve", "m", "--parallel", "2"],
            vec!["nss", "serve", "m", "--backend", "some-program"],
            vec!["nss", "serve", "--", "-hf", "org/model"],
            vec!["nss", "serve", "m", "--", "--future-flag"],
            vec!["nss", "serve", "a", "--model", "b"],
            vec!["nss", "serve", "-o"],
            vec!["nss", "serve", "-o", ""],
            vec!["nss", "serve", "-o", "llama3.2", "-od"],
            vec!["nss", "serve", "-o", "llama3.2", "-od="],
            vec!["nss", "serve", "a.gguf", "-o", "llama3.2"],
            vec!["nss", "serve", "-o", "llama3.2", "--ollama", "other"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
