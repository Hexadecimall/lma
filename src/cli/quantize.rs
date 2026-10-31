use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use super::{
    Action, Syntax,
    args::{Args, Flag, threads},
    help, unsupported,
};
use crate::config::{Job, Plan, QUANT_TYPES, QuantizeConfig, default_threads};

pub(super) fn parse(mut args: Args, mut syntax: Syntax) -> Result<Action, String> {
    let mut long_options = false;
    let mut positional = Vec::new();
    let mut output = None;
    let mut kind = None;
    let mut thread_count = None;
    let mut pure = false;
    let mut leave_output = false;
    let mut allow_requantize = false;
    let mut force = false;
    let mut dry_run = false;
    while let Some(argument) = args.next() {
        if argument == "--" {
            long_options = true;
            syntax = Syntax::Nss;
            continue;
        }
        let Some(flag) = Flag::parse(&argument) else {
            positional.push(argument);
            continue;
        };
        let friendly = !long_options;
        let name = match flag.name.as_str() {
            "-h" if friendly => "--help",
            "-q" if friendly => "--type",
            "-o" if friendly => "--out",
            "-t" if friendly => "--threads",
            "--leave-output" if friendly => "--leave-output-tensor",
            other => other,
        };
        match name {
            "--help" | "--help-all" => {
                flag.switch()?;
                return Ok(Action::Help(help::QUANTIZE));
            }
            "--type" => {
                if kind
                    .replace(args.text(&flag)?.to_ascii_uppercase())
                    .is_some()
                {
                    return Err("type specified twice".into());
                }
            }
            "--out" => {
                if output.replace(args.value(&flag)?).is_some() {
                    return Err("output specified twice".into());
                }
            }
            "--threads" => {
                if thread_count.replace(threads(&args.text(&flag)?)?).is_some() {
                    return Err("threads specified twice".into());
                }
            }
            "--pure" => {
                flag.switch()?;
                pure = true;
            }
            "--leave-output-tensor" => {
                flag.switch()?;
                leave_output = true;
            }
            "--allow-requantize" => {
                flag.switch()?;
                allow_requantize = true;
            }
            "--force" => {
                flag.switch()?;
                force = true;
            }
            "--dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            _ => return Err(unsupported(&flag.name)),
        }
    }
    let max = if syntax == Syntax::Nss { 4 } else { 2 };
    if positional.is_empty() || positional.len() > max {
        return Err(
            "use `lma q INPUT [OUTPUT] -t TYPE` or `lma nss quantize INPUT OUTPUT TYPE [THREADS]`"
                .into(),
        );
    }
    if positional.len() >= 2 && output.is_some() {
        return Err("output specified twice".into());
    }
    if let Some(value) = positional.get(2) {
        if kind.is_some() {
            return Err("type specified twice".into());
        }
        kind = Some(
            value
                .to_str()
                .ok_or("type must be UTF-8")?
                .to_ascii_uppercase(),
        );
    }
    if let Some(value) = positional.get(3) {
        if thread_count.is_some() {
            return Err("threads specified twice".into());
        }
        thread_count = Some(threads(value.to_str().ok_or("threads must be UTF-8")?)?);
    }
    let kind = kind.unwrap_or_else(|| "Q4_K_M".into());
    if !QUANT_TYPES.contains(&kind.as_str()) {
        return Err(format!(
            "unsupported quantization type '{kind}'; run `lma h q` for supported types"
        ));
    }
    let input = PathBuf::from(&positional[0]);
    if input.as_os_str().is_empty() {
        return Err("input cannot be empty".into());
    }
    let output = output
        .or_else(|| positional.get(1).cloned())
        .map(PathBuf::from)
        .map_or_else(|| default_output(&input, &kind), Ok)?;
    if output.as_os_str().is_empty() {
        return Err("output cannot be empty".into());
    }
    let threads = thread_count
        .map(|s| s.parse::<i32>().map_err(|_| "thread count is too large"))
        .transpose()?
        .unwrap_or_else(default_threads);
    Ok(Action::Run(Box::new(Plan {
        job: Job::Quantize(QuantizeConfig {
            input,
            output,
            kind,
            threads,
            pure,
            leave_output,
            allow_requantize,
            force,
        }),
        dry_run,
    })))
}

fn default_output(input: &Path, kind: &str) -> Result<PathBuf, String> {
    let filename = input.file_name().ok_or("input must name a model file")?;
    let stem = if input
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        input.file_stem().unwrap_or(filename)
    } else {
        filename
    };
    let mut name = OsString::from(stem);
    name.push(format!(".{kind}.gguf"));
    Ok(input.with_file_name(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;

    #[test]
    fn default_output_and_type_are_preserved() {
        let Job::Quantize(config) = plan(&["q", "models/my model.gguf"]).job else {
            panic!()
        };
        assert_eq!(config.output, Path::new("models/my model.Q4_K_M.gguf"));
        assert_eq!(config.kind, "Q4_K_M");
        assert!(!config.force);
    }

    #[test]
    fn flags_populate_quantization_parameters() {
        let Job::Quantize(config) = plan(&[
            "q", "in.gguf", "out.gguf", "-t", "q8_0", "-ts", "4", "-p", "-l", "-a", "-f",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.kind, "Q8_0");
        assert_eq!(config.threads, 4);
        assert!(config.pure && config.leave_output && config.allow_requantize && config.force);
    }

    #[test]
    fn nss_keeps_quant_alias_and_legacy_option_meanings() {
        for command in ["quantize", "quant"] {
            let parsed = plan(&[
                "nss",
                command,
                "in.gguf",
                "-o",
                "out.gguf",
                "-q",
                "q8_0",
                "-t",
                "4",
                "--pure",
                "--leave-output",
                "--allow-requantize",
                "--force",
                "--dry-run",
            ]);
            assert!(parsed.dry_run);
            let Job::Quantize(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.input, Path::new("in.gguf"));
            assert_eq!(config.output, Path::new("out.gguf"));
            assert_eq!(config.kind, "Q8_0");
            assert_eq!(config.threads, 4);
            assert!(config.pure && config.leave_output && config.allow_requantize && config.force);
        }
    }

    #[test]
    fn unknown_types_and_flags_are_rejected() {
        for argv in [
            vec!["q"],
            vec!["nss", "quantize", "in", "--type", "imaginary"],
            vec!["nss", "quantize", "in", "--importance", "matrix"],
            vec!["nss", "quantize", "in", "out", "--out", "other"],
            vec!["nss", "quantize", "in", "--", "--keep-split"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
