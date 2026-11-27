//! Starting the Python engine: one `python -m jaull <command>` process per action.

use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// How to start Jaull: JAULL_PYTHON (an interpreter with Jaull installed), then the Python
/// the installer puts next to this executable, then this checkout's virtualenv, then a
/// `jaull` on PATH. Python is started directly, with `-m jaull`, so that cancelling a
/// search can end the whole process tree.
pub fn command() -> Command {
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("python").join("python.exe")))
        .filter(|python| python.exists())
        .map(|python| python.to_string_lossy().into_owned());
    let python = std::env::var("JAULL_PYTHON").ok().or(bundled).or_else(|| {
        let dev = concat!(env!("CARGO_MANIFEST_DIR"), "/../.venv/Scripts/python.exe");
        Path::new(dev).exists().then(|| dev.to_string())
    });
    let mut cmd = match python {
        Some(python) => {
            let mut cmd = Command::new(python);
            cmd.args(["-m", "jaull"]);
            cmd
        }
        None => Command::new("jaull"),
    };
    hide_console(&mut cmd);
    cmd
}

/// Ends a process and its children (the virtualenv's python.exe starts another one).
pub fn kill_tree(pid: u32) {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/F", "/T", "/PID", &pid.to_string()]).stdout(Stdio::null()).stderr(Stdio::null());
    hide_console(&mut cmd);
    let _ = cmd.status();
}

/// Runs a command that prints one JSON document and waits for it.
pub fn run_json(args: &[&str]) -> Result<Value, String> {
    let out = command()
        .args(args)
        .output()
        .map_err(|e| format!("Could not start Jaull: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("Jaull failed: {}", err.trim()));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("Unexpected output from Jaull: {e}"))
}

/// `jaull serve stop`, waiting for it; true when it worked.
pub fn serve_stop() -> bool {
    command()
        .args(["serve", "stop"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Removes terminal colour codes, in case a tool prints them anyway.
pub fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Without this every Jaull process would flash a console window.
fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}
