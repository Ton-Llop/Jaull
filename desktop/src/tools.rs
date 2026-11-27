//! Advanced tools: `jaull inspect|estimate|doctor`, shown as their plain-text report.

use slint::ComponentHandle;

use crate::{jaull, AppWindow};

pub fn run_tool(window: &AppWindow, tool: i32, repo: &str, quant: &str, context: &str) {
    let mut cmd = jaull::command();
    match tool {
        1 => {
            cmd.args(["inspect", repo.trim()]);
        }
        2 => {
            cmd.args(["estimate", repo.trim()]);
            if !quant.trim().is_empty() {
                cmd.args(["-q", quant.trim()]);
            }
            if !context.trim().is_empty() {
                cmd.args(["-c", context.trim()]);
            }
        }
        _ => {
            cmd.arg("doctor");
        }
    }
    cmd.env("NO_COLOR", "1").env("COLUMNS", "100").env("TERM", "dumb");
    window.set_tool_running(true);
    window.set_tool_output("".into());
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let output = match cmd.output() {
            Ok(out) => {
                let mut text = String::from_utf8_lossy(&out.stdout).to_string();
                let err = String::from_utf8_lossy(&out.stderr);
                if !err.trim().is_empty() {
                    text.push('\n');
                    text.push_str(&err);
                }
                jaull::strip_ansi(&text)
            }
            Err(e) => format!("Could not start Jaull: {e}"),
        };
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(window) = weak.upgrade() {
                window.set_tool_running(false);
                window.set_tool_output(output.trim_end().into());
            }
        });
    });
}
