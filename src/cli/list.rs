use std::path::PathBuf;

use super::{
    Action,
    args::{Args, Flag},
    help,
};
use crate::config::{Job, ListConfig, Plan};

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut config = ListConfig::default();
    let mut filter = None;
    let mut dry_run = false;
    let mut positional = false;
    while let Some(argument) = args.next() {
        if !positional && argument == "--" {
            positional = true;
            continue;
        }
        if !positional && let Some(flag) = Flag::parse(&argument) {
            match flag.name.as_str() {
                "-h" | "--help" | "--help-all" => {
                    flag.switch()?;
                    return Ok(Action::Help(help::LIST));
                }
                "--json" => {
                    flag.switch()?;
                    config.json = true;
                }
                "--dir" => config.local_dirs.push(PathBuf::from(args.value(&flag)?)),
                "-od" | "--ollama-dir" => {
                    config.ollama_dir = Some(PathBuf::from(args.value(&flag)?))
                }
                "--dry-run" => {
                    flag.switch()?;
                    dry_run = true;
                }
                _ => {
                    return Err(format!(
                        "unknown list option '{}'; run `lma h l`",
                        flag.name
                    ));
                }
            }
        } else if filter
            .replace(
                argument
                    .into_string()
                    .map_err(|_| "model filter must be UTF-8")?,
            )
            .is_some()
        {
            return Err(
                "list accepts one optional model filter; quote filters containing spaces".into(),
            );
        }
    }
    if config.local_dirs.len() > 120 {
        return Err("list accepts at most 120 --dir options".into());
    }
    config.filter = filter.unwrap_or_default();
    if config.filter.len() > 512 || config.filter.chars().any(char::is_control) {
        return Err("model filter must be at most 512 bytes without control characters".into());
    }
    Ok(Action::Run(Box::new(Plan {
        job: Job::List(config),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    #[test]
    fn list_owns_l_alias_and_preserves_nss_option_shortcuts() {
        for argv in [
            vec![
                "l",
                "pulsar",
                "-dr=local",
                "-dr",
                "other",
                "-o",
                "store",
                "-j",
                "-d",
            ],
            vec![
                "nss",
                "list",
                "pulsar",
                "--dir=local",
                "--dir",
                "other",
                "-od",
                "store",
                "--json",
                "--dry-run",
            ],
            vec![
                "nss",
                "list",
                "pulsar",
                "--dir=local",
                "--dir",
                "other",
                "--ollama-dir=store",
                "--json",
                "--dry-run",
            ],
        ] {
            let plan = plan(&argv);
            assert!(plan.dry_run);
            let Job::List(config) = plan.job else {
                panic!()
            };
            assert_eq!(config.filter, "pulsar");
            assert_eq!(
                config.local_dirs,
                vec![PathBuf::from("local"), PathBuf::from("other")]
            );
            assert_eq!(config.ollama_dir, Some(PathBuf::from("store")));
            assert!(config.json);
        }
        assert!(matches!(plan(&["lh", "-d"]).job, Job::Launch(_)));
    }

    #[test]
    fn nss_terminator_preserves_dash_prefixed_filters() {
        let Job::List(config) = plan(&["nss", "list", "--json", "--", "--help"]).job else {
            panic!()
        };
        assert!(config.json);
        assert_eq!(config.filter, "--help");
        assert!(crate::cli::parse(["l", "--", "--help"].map(OsString::from)).is_err());
    }

    #[test]
    fn invalid_list_options_are_not_ignored() {
        for argv in [
            vec!["nss", "l"],
            vec!["nss", "list", "--json=true"],
            vec!["nss", "list", "--dir"],
            vec!["nss", "list", "--ollama-dir"],
            vec!["l", "one", "two"],
            vec!["nss", "list", "--mode", "chat"],
            vec!["l", "bad\nfilter"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
