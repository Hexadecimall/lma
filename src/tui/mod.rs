//! Native terminal workspace over the same in-process engine as the desktop host.
//! No network listener is opened for chat, and no external client is launched.
mod attachments;
mod chat;
mod editor;
mod logs;
mod markdown;
mod setup;
mod state;
mod stream;
mod terminal;
mod text;
mod view;

use crate::config::{Job, LaunchConfig, Plan};

/// Called synchronously by the root runtime, before entering any Tokio runtime.
pub fn run(config: LaunchConfig) -> Result<(), String> {
    run_inner(config).map_err(|error| text::sanitize(&error))
}

fn run_inner(mut config: LaunchConfig) -> Result<(), String> {
    if !matches!(config.mode.as_str(), "chat" | "host" | "launch") {
        return Err("launch mode must be chat, host, or launch".into());
    }
    loop {
        let needs_setup = config.interactive || config.serve.model.as_os_str().is_empty();
        // Explicit hosting is usable in scripts, with no terminal initialization.
        if config.mode == "host" && !needs_setup {
            return host(config);
        }
        if config.mode == "launch"
            && config.target != crate::config::LaunchTarget::Chat
            && !needs_setup
        {
            return crate::runtime::launch::run(config.serve, config.target);
        }
        terminal::check_tty()?;
        if needs_setup {
            let Some(selected) = setup::run(config)? else {
                return Ok(());
            };
            config = selected;
            config.interactive = false;
        }
        if config.mode == "host" {
            return host(config);
        }
        if config.mode == "launch" && config.target != crate::config::LaunchTarget::Chat {
            return crate::runtime::launch::run(config.serve, config.target);
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("cannot start terminal async runtime: {e}"))?;
        // Client presets are host connection profiles, not system prompts or agents.
        let mut serve = config.serve.clone();
        serve.api = "all".into();
        serve.client = "default".into();
        let vision = cfg!(feature = "vision")
            && (serve.projector.is_some()
                || crate::models::resolve(&serve.model, serve.ollama_dir.as_deref())
                    .is_ok_and(|model| model.projector.is_some()));
        eprintln!(
            "Loading {} for local chat…",
            text::one_line(&serve.model.to_string_lossy())
        );
        let native_logs = logs::NativeLogs::capture();
        let result = runtime.block_on(async {
            // Load before entering the alternate screen: engine placement/fallback
            // diagnostics printed by the core remain visible in terminal scrollback.
            let session = crate::desktop::start(serve.clone()).await?;
            chat::run(session, serve, vision).await
        });
        let recent = logs::recent();
        drop(native_logs);
        match result {
            Ok(chat::Outcome::Quit) => return Ok(()),
            Ok(chat::Outcome::Models) => config.interactive = true,
            Err(error) => {
                return Err(if recent.is_empty() {
                    error
                } else {
                    format!("{error}\nRecent native diagnostics:\n{recent}")
                });
            }
        }
    }
}

fn host(config: LaunchConfig) -> Result<(), String> {
    crate::runtime::run(Plan {
        job: Job::Serve(config.serve),
        dry_run: false,
    })
}
