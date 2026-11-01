use super::{
    Action,
    args::{Args, Flag},
    help,
};
use crate::config::{Job, Plan, UpdateConfig};

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut config = UpdateConfig {
        check: false,
        version: None,
    };
    let mut dry_run = false;
    while let Some(argument) = args.next() {
        let flag = Flag::parse(&argument)
            .ok_or("update takes options only; use -v VERSION to select a release")?;
        match flag.name.as_str() {
            "--help" | "--help-all" | "-h" => {
                flag.switch()?;
                return Ok(Action::Help(help::UPDATE));
            }
            "--check" => {
                flag.switch()?;
                config.check = true;
            }
            "--version" => {
                let value = args.text(&flag)?;
                let version = value.strip_prefix('v').unwrap_or(&value);
                let parts: Vec<_> = version.split('.').collect();
                if parts.len() != 3
                    || parts.iter().any(|part| {
                        part.is_empty()
                            || (part.len() > 1 && part.starts_with('0'))
                            || !part.bytes().all(|b| b.is_ascii_digit())
                            || part.parse::<u32>().is_err()
                    })
                {
                    return Err("version must be a stable MAJOR.MINOR.PATCH release, optionally prefixed with v".into());
                }
                config.version = Some(format!("v{version}"));
            }
            "--dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            _ => {
                return Err(format!(
                    "unknown update option '{}'; run `lma h u`",
                    flag.name
                ));
            }
        }
    }
    Ok(Action::Run(Box::new(Plan {
        job: Job::Update(config),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    #[test]
    fn root_and_nss_update_options_match_without_network() {
        for argv in [
            vec!["u", "-c", "-v", "0.3.4", "-d"],
            vec!["nss", "update", "--check", "--version=v0.3.4", "--dry-run"],
        ] {
            let plan = plan(&argv);
            assert!(plan.dry_run);
            let Job::Update(config) = plan.job else {
                panic!()
            };
            assert!(config.check);
            assert_eq!(config.version.as_deref(), Some("v0.3.4"));
        }
        let Job::Update(config) = plan(&["u"]).job else {
            panic!()
        };
        assert!(!config.check);
        assert!(config.version.is_none());
    }

    #[test]
    fn update_rejects_ambiguous_or_unsafe_release_selection() {
        for argv in [
            vec!["u", "v0.3.4"],
            vec!["u", "-v", "../bad"],
            vec!["u", "-v", "0.3"],
            vec!["u", "-v", "0.3.4-beta"],
            vec!["u", "-v", "00.3.4"],
            vec!["u", "-v", "+1.2.3"],
            vec!["u", "-v", "1.2.4294967296"],
            vec!["u", "-v", "1.2.3+build"],
            vec!["u", "-v", "1.2.3.4"],
            vec!["u", "-v", " 1.2.3"],
            vec!["u", "-c=yes"],
            vec!["u", "-v"],
            vec!["u", "--check"],
            vec!["u", "--force"],
            vec!["u", "--"],
        ] {
            assert!(
                crate::cli::parse(argv.iter().map(OsString::from)).is_err(),
                "{argv:?}"
            );
        }
    }
}
