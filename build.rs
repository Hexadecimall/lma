//! Standalone build-time launcher: `rustc --edition=2024 build.rs -o lma-build`.
//! Cargo auto-discovery is disabled with `package.build = false`.
//! The application itself never launches these compiler processes.

use std::{
    env,
    ffi::OsString,
    path::Path,
    process::{Command, ExitCode},
};

const HELP: &str = "lma optimized build

SETUP (from the repository root)
  rustc --edition=2024 build.rs -o lma-build
  ./lma-build [OPTIONS]

OPTIONS
  --cpu          No explicit GPU feature (Apple Silicon still includes Metal)
  --metal        Metal backend (macOS only; default on macOS)
  --cuda         NVIDIA CUDA backend (requires the CUDA toolkit)
  --portable     Do not add host-CPU tuning; keep any supplied Rust flags
  -j, --jobs N   Maximum Cargo build jobs
  --dry-run      Print the build command without compiling lma
  -h, --help     Show this help

Always includes vision, image generation, fetch, quantization, and browser chat.
Outside macOS the default backend is CPU; CUDA is explicitly opt-in.
Apple Silicon includes llama.cpp Metal even with --cpu; run `lma s MODEL -g cpu`.
Serving defaults to all GPU layers with compiled Metal/CUDA, otherwise CPU.
Builds for the active Rust compiler's host, not a cross-compilation target.
Release settings: opt-level 3, fat LTO, one codegen unit, stripped symbols.
Host-tuned binaries may not run on older CPUs; use --portable for distribution.
Fat LTO increases build time and peak memory; try -j 2 on smaller machines.
";

#[derive(Debug, PartialEq)]
struct Options {
    backend: &'static str,
    native: bool,
    jobs: Option<usize>,
    dry_run: bool,
}

fn parse(args: impl IntoIterator<Item = String>, macos: bool) -> Result<Option<Options>, String> {
    let mut options = Options {
        backend: if macos { "metal" } else { "cpu" },
        native: true,
        jobs: None,
        dry_run: false,
    };
    let mut backend = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--cpu" | "--metal" | "--cuda" => {
                let selected = match arg.as_str() {
                    "--metal" => "metal",
                    "--cuda" => "cuda",
                    _ => "cpu",
                };
                if backend.replace(selected).is_some() {
                    return Err("choose only one backend: --cpu, --metal, or --cuda".into());
                }
                options.backend = selected;
            }
            "--portable" => options.native = false,
            "--dry-run" => options.dry_run = true,
            "-j" | "--jobs" => {
                options.jobs = Some(
                    args.next()
                        .and_then(|s| s.parse::<usize>().ok())
                        .filter(|n| *n > 0)
                        .ok_or("--jobs needs a positive integer")?,
                );
            }
            _ => return Err(format!("unknown build option {arg:?}; use --help")),
        }
    }
    if options.backend == "metal" && !macos {
        return Err("Metal requires a macOS host; use --cpu or --cuda".into());
    }
    if options.backend == "cuda" && macos {
        return Err("CUDA is not supported on macOS; use --metal or --cpu".into());
    }
    Ok(Some(options))
}

fn rustflags(
    encoded: Option<String>,
    plain: Option<String>,
    native: bool,
) -> Result<String, String> {
    // Match Cargo's precedence and its whitespace splitting for plain RUSTFLAGS.
    let mut flags = encoded.unwrap_or_else(|| {
        plain
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("\x1f")
    });
    if native {
        if flags.contains("target-cpu") {
            return Err("existing Rust flags specify target-cpu; use --portable to keep your custom CPU tuning".into());
        }
        if !flags.is_empty() {
            flags.push('\x1f');
        }
        flags.push_str("-C\x1ftarget-cpu=native");
    }
    Ok(flags)
}

fn host_triple(version: &str) -> Result<&str, String> {
    version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .filter(|host| !host.is_empty() && !host.chars().any(char::is_whitespace))
        .ok_or_else(|| "rustc did not report its host target".into())
}

fn build_command(options: &Options, cargo: OsString, host: &str, flags: String) -> Command {
    let features = if options.backend == "cpu" {
        "vision,image-gen,speech".to_owned()
    } else {
        format!("vision,image-gen,speech,{}", options.backend)
    };
    let mut command = Command::new(cargo);
    command.args([
        "build",
        "--locked",
        "--release",
        "--bin",
        "lma",
        "--no-default-features",
        "--features",
        &features,
        // An explicit host target prevents Cargo config/environment from making
        // target-cpu=native unsafe through accidental cross-compilation.
        "--target",
        host,
    ]);
    if let Some(jobs) = options.jobs {
        command.args(["--jobs", &jobs.to_string()]);
    }
    command.env("CARGO_ENCODED_RUSTFLAGS", flags);
    // The pinned llama-cpp-sys-2 consumes these settings and translates
    // target-cpu=native into GGML_NATIVE=ON for its C/C++ build as well.
    command.env("LLAMA_LIB_PROFILE", "Release");
    command.env("CMAKE_BUILD_TYPE", "Release");
    command
}

fn run() -> Result<ExitCode, String> {
    let Some(options) = parse(env::args().skip(1), cfg!(target_os = "macos"))? else {
        println!("{HELP}");
        return Ok(ExitCode::SUCCESS);
    };
    if !Path::new("Cargo.toml").is_file() || !Path::new("src/main.rs").is_file() {
        return Err("run lma-build from the lma repository root".into());
    }
    let flags = rustflags(
        env::var("CARGO_ENCODED_RUSTFLAGS").ok(),
        env::var("RUSTFLAGS").ok(),
        options.native,
    )?;
    let compiler = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = Command::new(compiler)
        .args(["--version", "--verbose"])
        .output()
        .map_err(|e| format!("cannot query rustc: {e}"))?;
    if !version.status.success() {
        return Err(format!(
            "rustc host lookup failed: {}",
            String::from_utf8_lossy(&version.stderr)
        ));
    }
    let version =
        String::from_utf8(version.stdout).map_err(|e| format!("invalid rustc version: {e}"))?;
    let host = host_triple(&version)?;
    let mut command = build_command(
        &options,
        env::var_os("CARGO").unwrap_or_else(|| "cargo".into()),
        host,
        flags,
    );
    println!(
        "Building all app features with {} feature selection for {host} (Apple Silicon retains Metal).",
        options.backend
    );
    println!("{command:?}");
    if options.dry_run {
        return Ok(ExitCode::SUCCESS);
    }
    let status = command
        .status()
        .map_err(|e| format!("cannot start Cargo: {e}"))?;
    if !status.success() {
        return Err(format!(
            "Cargo build failed ({status}); see compiler output above"
        ));
    }
    println!("Build complete. Binary is under <Cargo target directory>/{host}/release/.");
    println!(
        "Serving defaults to all GPU layers with compiled Metal/CUDA (including implicit Apple Silicon Metal), otherwise CPU. Use `lma s MODEL -g cpu` to disable all inference offload."
    );
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("build error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(args: &[&str], macos: bool) -> Result<Option<Options>, String> {
        parse(args.iter().map(|s| s.to_string()), macos)
    }

    #[test]
    fn defaults_and_backend_validation() {
        assert_eq!(options(&[], true).unwrap().unwrap().backend, "metal");
        assert_eq!(options(&[], false).unwrap().unwrap().backend, "cpu");
        assert_eq!(
            options(&["--cuda"], false).unwrap().unwrap().backend,
            "cuda"
        );
        for (args, macos) in [
            (vec!["--metal", "--cpu"], true),
            (vec!["--cuda"], true),
            (vec!["--metal"], false),
            (vec!["--all-features"], false),
            (vec!["--jobs", "0"], false),
            (vec!["-j"], false),
        ] {
            assert!(options(&args, macos).is_err());
        }
    }

    #[test]
    fn help_and_build_controls() {
        assert!(options(&["--help"], false).unwrap().is_none());
        let parsed = options(&["--portable", "--dry-run", "-j", "2"], true)
            .unwrap()
            .unwrap();
        assert!(!parsed.native);
        assert!(parsed.dry_run);
        assert_eq!(parsed.jobs, Some(2));
    }

    #[test]
    fn flags_preserve_cargo_precedence_and_detect_conflicting_cpu_tuning() {
        assert_eq!(
            rustflags(None, None, true).unwrap(),
            "-C\x1ftarget-cpu=native"
        );
        assert_eq!(
            rustflags(Some("-Dwarnings".into()), Some("ignored".into()), true).unwrap(),
            "-Dwarnings\x1f-C\x1ftarget-cpu=native"
        );
        assert_eq!(
            rustflags(None, Some("-C target-cpu=generic".into()), false).unwrap(),
            "-C\x1ftarget-cpu=generic"
        );
        assert!(rustflags(None, Some("-C target-cpu=generic".into()), true).is_err());
        assert_eq!(rustflags(None, None, false).unwrap(), "");
    }

    #[test]
    fn compiler_host_is_required() {
        assert_eq!(
            host_triple("rustc 1.0\nhost: aarch64-apple-darwin\n").unwrap(),
            "aarch64-apple-darwin"
        );
        assert_eq!(
            host_triple("rustc 1.0\r\nhost: x86_64-pc-windows-msvc\r\n").unwrap(),
            "x86_64-pc-windows-msvc"
        );
        for version in [
            "",
            "rustc 1.0",
            "host: ",
            "host: invalid target",
            "host: target\t",
            "host:target",
        ] {
            assert!(host_triple(version).is_err(), "accepted {version:?}");
        }
    }

    #[test]
    fn build_arguments_and_environment() {
        use std::ffi::OsStr;

        for (args, macos, host, features) in [
            (
                vec![],
                true,
                "aarch64-apple-darwin",
                "vision,image-gen,speech,metal",
            ),
            (
                vec![],
                false,
                "x86_64-unknown-linux-gnu",
                "vision,image-gen,speech",
            ),
            (
                vec!["--cpu", "-j", "2"],
                true,
                "aarch64-apple-darwin",
                "vision,image-gen,speech",
            ),
            (
                vec!["--cuda", "--portable"],
                false,
                "x86_64-unknown-linux-gnu",
                "vision,image-gen,speech,cuda",
            ),
        ] {
            let options = options(&args, macos).unwrap().unwrap();
            let flags = rustflags(None, Some("-D warnings".into()), options.native).unwrap();
            let command = build_command(&options, "custom-cargo".into(), host, flags.clone());
            assert_eq!(command.get_program(), OsStr::new("custom-cargo"));
            let mut expected = vec![
                "build",
                "--locked",
                "--release",
                "--bin",
                "lma",
                "--no-default-features",
                "--features",
                features,
                "--target",
                host,
            ];
            if options.jobs.is_some() {
                expected.extend(["--jobs", "2"]);
            }
            assert_eq!(
                command.get_args().collect::<Vec<_>>(),
                expected.iter().map(OsStr::new).collect::<Vec<_>>()
            );
            let environment = command
                .get_envs()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(environment.len(), 3);
            assert_eq!(
                environment[OsStr::new("CARGO_ENCODED_RUSTFLAGS")],
                Some(OsStr::new(&flags))
            );
            assert_eq!(
                environment[OsStr::new("LLAMA_LIB_PROFILE")],
                Some(OsStr::new("Release"))
            );
            assert_eq!(
                environment[OsStr::new("CMAKE_BUILD_TYPE")],
                Some(OsStr::new("Release"))
            );
        }
    }

    #[test]
    fn portable_flags_preserve_encoded_values_even_when_empty() {
        assert_eq!(
            rustflags(Some("".into()), Some("-D warnings".into()), false).unwrap(),
            ""
        );
        assert_eq!(
            rustflags(
                Some("-C\x1ftarget-cpu=generic".into()),
                Some("ignored".into()),
                false
            )
            .unwrap(),
            "-C\x1ftarget-cpu=generic"
        );
        assert!(rustflags(Some("-Ctarget-cpu=generic".into()), None, true).is_err());
    }
}
