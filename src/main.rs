use lma::{cli, runtime, terminal_style::TerminalStyle};

use std::process::ExitCode;

// main function
fn main() -> ExitCode {
    // here we parse all of the arguemments and get the command (or error if there is no command)
    let action = match cli::parse(std::env::args_os().skip(1)) {
        Ok(action) => action,
        Err(error) => {
            let style = TerminalStyle::stderr();
            eprintln!("{}\n\nRun `lma h` for usage.", style.error(&error));
            return ExitCode::from(2);
        }
    };
    // simple match thingy for the commands
    match action {
        cli::Action::Help(help) => {
            print!("{}", TerminalStyle::stdout().help(help));
            ExitCode::SUCCESS
        }
        cli::Action::Version => {
            println!(
                "{}",
                TerminalStyle::stdout().version(env!("CARGO_PKG_VERSION"))
            );
            ExitCode::SUCCESS
        }
        cli::Action::Run(plan) => match runtime::run(*plan) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{}", TerminalStyle::stderr().error(&error));
                ExitCode::FAILURE
            }
        },
    }
}
