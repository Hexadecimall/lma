use std::collections::HashSet;

use super::{
    Action,
    args::{Args, Flag, number},
    help,
};
use crate::config::{Job, Plan, StopConfig};

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut config = StopConfig::default();
    let mut dry_run = false;
    let mut seen = HashSet::new();
    while let Some(argument) = args.next() {
        let Some(flag) = Flag::parse(&argument) else {
            return Err("stop accepts an optional -m/--model NAME, not positional arguments; use `lma s -p FILE` or `lma nss serve --path FILE` to serve a file".into());
        };
        let name = match flag.name.as_str() {
            "-h" | "--help" | "--help-all" => {
                flag.switch()?;
                return Ok(Action::Help(help::STOP));
            }
            "-m" | "--model" => "model",
            "--host" => "host",
            "--port" => "port",
            "--api-key" => "api-key",
            "--dry-run" => "dry-run",
            _ => {
                return Err(format!(
                    "unknown stop option '{}'; run `lma h sp`",
                    flag.name
                ));
            }
        };
        if name != "dry-run" && !seen.insert(name) {
            return Err(format!("stop {name} specified twice"));
        }
        match name {
            "model" => {
                let model = args.text(&flag)?;
                if model.len() > 512 || model.chars().any(char::is_control) {
                    return Err("stop model must be 1..512 bytes without control characters".into());
                }
                config.model = Some(model);
            }
            "host" => {
                config.host = args.text(&flag)?;
                if config.host.chars().any(char::is_whitespace) {
                    return Err("stop host cannot contain whitespace".into());
                }
            }
            "port" => {
                config.port = u16::try_from(number(&args.text(&flag)?, "--port", false)?)
                    .map_err(|_| "port must be between 1 and 65535")?;
            }
            "api-key" => {
                let key = args.text(&flag)?;
                if key.chars().any(char::is_whitespace) {
                    return Err("API key cannot contain whitespace".into());
                }
                config.api_key = Some(key);
            }
            "dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            _ => unreachable!(),
        }
    }
    Ok(Action::Run(Box::new(Plan {
        job: Job::Stop(config),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use crate::{cli::tests::plan, config::Job};
    use std::ffi::OsString;

    #[test]
    fn stop_is_distinct_from_serving_and_defaults_to_local_managed_host() {
        for argv in [vec!["sp", "-d"], vec!["nss", "stop", "--dry-run"]] {
            let parsed = plan(&argv);
            assert!(parsed.dry_run);
            let Job::Stop(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.model, None);
            assert_eq!(config.host, "127.0.0.1");
            assert_eq!(config.port, 8080);
            assert_eq!(config.api_key, None);
        }
    }

    #[test]
    fn stop_options_support_nss_and_inline_values() {
        for argv in [
            vec![
                "sp",
                "-m=local:latest",
                "-ht=localhost",
                "-p=9000",
                "-a=secret",
            ],
            vec![
                "nss",
                "--dry-run",
                "stop",
                "-m=local:latest",
                "--host=localhost",
                "--port=9000",
                "--api-key=secret",
            ],
        ] {
            let Job::Stop(config) = plan(&argv).job else {
                panic!()
            };
            assert_eq!(config.model.as_deref(), Some("local:latest"));
            assert_eq!(config.host, "localhost");
            assert_eq!(config.port, 9000);
            let json = serde_json::to_string(&config).unwrap();
            assert!(!json.contains("secret"));
            assert!(json.contains("[redacted]"));
        }
    }

    #[test]
    fn stop_rejects_legacy_serve_syntax_and_invalid_values() {
        for flags in [
            vec!["model.gguf"],
            vec!["--path", "model.gguf"],
            vec!["--gpu", "cpu"],
            vec!["-m"],
            vec!["--model="],
            vec!["-m", "a", "--model", "b"],
            vec!["--port", "0"],
            vec!["--port", "65536"],
            vec!["--port", "8k"],
            vec!["--host="],
            vec!["--host", "local host"],
            vec!["--api-key", "bad key"],
            vec!["--dry-run=true"],
        ] {
            let mut argv = vec!["nss", "stop"];
            argv.extend(flags);
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
