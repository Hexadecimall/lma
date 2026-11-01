use super::{
    Action, Syntax,
    args::{Args, Flag},
    help, nss_command_named, parse_command,
};

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut dry_run = false;
    while let Some(argument) = args.next() {
        if argument == "help" {
            return help::parse(args, Syntax::Nss);
        }
        if let Some(command) = nss_command_named(&argument) {
            let mut arguments = args.rest();
            if dry_run {
                arguments.insert(0, "--dry-run".into());
            }
            return parse_command(command, Args::new(arguments), Syntax::Nss);
        }
        if let Some(flag) = Flag::parse(&argument) {
            match flag.name.as_str() {
                "--help" | "-h" => {
                    flag.switch()?;
                    return Ok(Action::Help(help::NSS));
                }
                "--version" | "-V" | "-v" if args.is_empty() => {
                    flag.switch()?;
                    return Ok(Action::Version);
                }
                "--dry-run" => {
                    flag.switch()?;
                    dry_run = true;
                    continue;
                }
                _ => {}
            }
        }
        return Err("use `lma nss [--dry-run] serve|stop|launch|list|chat|transcribe|help|quantize|fetch|import|create|delete|update|image ...` or `lma h nss`; root short commands are disabled in nss (except the legacy lh alias)".into());
    }
    if dry_run {
        Err(
            "nss needs a command: serve, stop, launch, list, chat, transcribe, help, quantize, fetch, import, create, delete, update, or image"
                .into(),
        )
    } else {
        Ok(Action::Help(help::NSS))
    }
}

#[cfg(test)]
mod tests {
    use crate::{cli::tests::plan, config::Job};
    use std::ffi::OsString;

    #[test]
    fn long_form_server_options_use_the_same_embedded_engine() {
        let plan = plan(&[
            "nss",
            "--dry-run",
            "serve",
            "--model",
            "m.gguf",
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "0",
        ]);
        assert!(plan.dry_run);
        let Job::Serve(config) = plan.job else {
            panic!()
        };
        assert_eq!(config.context, 8192);
    }

    #[test]
    fn quantizer_positional_form_keeps_safety_checks() {
        let plan = plan(&["nss", "quantize", "in.gguf", "out.gguf", "q8_0", "4"]);
        let Job::Quantize(config) = plan.job else {
            panic!()
        };
        assert_eq!(config.kind, "Q8_0");
        assert_eq!(config.threads, 4);
        assert!(!config.force);
    }

    #[test]
    fn nss_root_help_and_version_do_not_require_a_command_or_model() {
        use crate::cli::{Action, parse};
        for flag in ["--help", "-h"] {
            assert!(matches!(
                parse(["nss", flag].map(OsString::from)).unwrap(),
                Action::Help(_)
            ));
        }
        for flag in ["--version", "-V", "-v"] {
            assert!(matches!(
                parse(["nss", flag].map(OsString::from)).unwrap(),
                Action::Version
            ));
            assert!(parse(["nss", flag, "serve"].map(OsString::from)).is_err());
        }
    }

    #[test]
    fn root_short_commands_and_unknown_flags_are_rejected() {
        for argv in [
            vec!["nss", "s", "-m", "m"],
            vec!["nss", "q", "in"],
            vec!["nss", "c", "-h"],
            vec!["nss", "t", "-h"],
            vec!["nss", "h"],
            vec!["nss", "sp"],
            vec!["nss", "ce"],
            vec!["nss", "it"],
            vec!["nss", "serve", "--model", "m", "--future-option"],
            vec!["nss", "--backend", "program", "serve"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
