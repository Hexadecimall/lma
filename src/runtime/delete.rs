//! Confirmation only: all model selection and filesystem safety live in models::delete.
use crate::config::DeleteConfig;
use std::io::{self, BufRead, IsTerminal, Write};

pub(super) fn run(config: DeleteConfig) -> Result<(), String> {
    let stdin = io::stdin();
    let stderr = io::stderr();
    confirm_deletion(
        &config,
        stdin.is_terminal(),
        &mut stdin.lock(),
        &mut stderr.lock(),
    )?;

    // Do not call a host's delete endpoint: it may unload an unrelated resident model.
    let result = crate::models::delete::run(
        &config.model,
        config.source.as_deref(),
        config.ollama_dir.as_deref(),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|error| format!("cannot serialize model deletion result: {error}"))?
    );
    Ok(())
}

fn confirm_deletion(
    config: &DeleteConfig,
    is_terminal: bool,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), String> {
    if config.yes {
        return Ok(());
    }
    if !is_terminal {
        return Err("model deletion requires terminal confirmation; unload the selected model first with `lma sp -m MODEL` or the desktop app, then pass -y (NSS: --yes) for non-interactive deletion; nothing was deleted".into());
    }
    write!(
        output,
        "Warning: unload the selected model first with `lma sp -m MODEL` or the desktop app, and stop its use in other processes. This CLI cannot verify other processes and will not unload anything. Deleting {:?} (source: {}) can permanently remove model files. Continue? [y/N] ",
        config.model,
        config.source.as_deref().unwrap_or("unspecified"),
    )
    .and_then(|()| output.flush())
    .map_err(|error| format!("cannot prompt for model deletion: {error}"))?;
    let mut reply = String::new();
    input.read_line(&mut reply).map_err(|error| {
        format!("cannot read model deletion confirmation: {error}; nothing was deleted")
    })?;
    if !affirmative(&reply) {
        return Err("model deletion cancelled; nothing was deleted".into());
    }
    Ok(())
}

fn affirmative(reply: &str) -> bool {
    let reply = reply.trim();
    reply.eq_ignore_ascii_case("y") || reply.eq_ignore_ascii_case("yes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn config(yes: bool) -> DeleteConfig {
        DeleteConfig {
            model: "fixture:latest".into(),
            source: Some("lma".into()),
            ollama_dir: None,
            yes,
        }
    }

    #[test]
    fn only_y_or_yes_is_affirmative() {
        for reply in ["y", "Y", "yes", "YES", "YeS", " yes \r\n"] {
            assert!(affirmative(reply), "{reply:?}");
        }
        for reply in [
            "",
            "\n",
            " ",
            "n",
            "no",
            "true",
            "1",
            "yeah",
            "y extra",
            "yes please",
            "yes\nno",
            "yes\0",
        ] {
            assert!(!affirmative(reply), "{reply:?}");
        }
    }

    #[test]
    fn non_terminal_refuses_even_a_piped_yes_without_reading_or_prompting() {
        let mut input = Cursor::new(b"yes\n");
        let mut output = Vec::new();
        let error = confirm_deletion(&config(false), false, &mut input, &mut output).unwrap_err();
        assert!(error.contains("--yes"));
        assert!(error.contains("nothing was deleted"));
        assert_eq!(input.position(), 0);
        assert!(output.is_empty());
    }

    #[test]
    fn explicit_yes_bypasses_all_confirmation_io() {
        for terminal in [false, true] {
            let mut input = Cursor::new(b"no\n");
            let mut output = Vec::new();
            confirm_deletion(&config(true), terminal, &mut input, &mut output).unwrap();
            assert_eq!(input.position(), 0);
            assert!(output.is_empty());
        }
    }

    #[test]
    fn terminal_prompt_warns_about_unloading_and_file_loss_before_acceptance() {
        for reply in ["y\n", "YES\n", " yes \r\n"] {
            let mut input = Cursor::new(reply.as_bytes());
            let mut output = Vec::new();
            confirm_deletion(&config(false), true, &mut input, &mut output).unwrap();
            let prompt = String::from_utf8(output).unwrap();
            for detail in [
                "lma sp -m MODEL",
                "desktop",
                "other processes",
                "will not unload",
                "permanently remove model files",
                "fixture:latest",
                "source: lma",
                "[y/N]",
            ] {
                assert!(prompt.contains(detail), "missing {detail}");
            }
        }
    }

    #[test]
    fn refusal_empty_response_and_eof_cancel() {
        for reply in ["", "\n", "n\n", "no\n", "sure\n", "yes please\n"] {
            let mut input = Cursor::new(reply.as_bytes());
            let mut output = Vec::new();
            let error =
                confirm_deletion(&config(false), true, &mut input, &mut output).unwrap_err();
            assert!(error.contains("cancelled"));
        }
    }

    #[test]
    fn confirmation_io_failures_are_errors() {
        struct BrokenIo;
        impl io::Read for BrokenIo {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("fixture read failure"))
            }
        }
        impl BufRead for BrokenIo {
            fn fill_buf(&mut self) -> io::Result<&[u8]> {
                Err(io::Error::other("fixture read failure"))
            }
            fn consume(&mut self, _: usize) {}
        }
        impl Write for BrokenIo {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("fixture write failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::other("fixture flush failure"))
            }
        }
        let mut output = Vec::new();
        let error = confirm_deletion(&config(false), true, &mut BrokenIo, &mut output).unwrap_err();
        assert!(error.contains("cannot read"));
        let error = confirm_deletion(
            &config(false),
            true,
            &mut Cursor::new(b"yes\n"),
            &mut BrokenIo,
        )
        .unwrap_err();
        assert!(error.contains("cannot prompt"));
        confirm_deletion(&config(true), false, &mut BrokenIo, &mut BrokenIo).unwrap();
    }
}
