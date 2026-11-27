//! The model server Jaull runs (llama-server): start, cancel, stop and install llama.cpp.

use std::io::{BufRead, BufReader};
use std::process::Stdio;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use slint::{ComponentHandle, ModelRc, VecModel};

use crate::format::{text, thousands};
use crate::{jaull, AppWindow, Msg};

/// Where the chat sends its requests.
#[derive(Clone)]
pub struct Endpoint {
    pub url: String,
    pub key: String,
    pub alias: String,
    /// Jaull's self-signed certificate when the server uses HTTPS (shared on the network).
    pub cert: Option<String>,
}

/// What the Chat page shows about the running server.
#[derive(Default)]
struct ServerInfo {
    alias: String,
    lan_url: String,
    firewall_rule: String,
    sharing: String,
}

/// The model server, as far as this window knows it.
#[derive(Default)]
pub struct Server {
    pub endpoint: Mutex<Option<Endpoint>>,
    /// Stop it when the window closes: only if this window started it.
    started_here: AtomicBool,
    /// Tokens of context each person has; 0 when Jaull did not say.
    pub context_per_user: AtomicU64,
    /// The conversation sent with every message (role, content).
    pub history: Mutex<Vec<(String, String)>>,
    /// The `serve start` still downloading or starting, so it can be cancelled.
    start_pid: Mutex<Option<u32>>,
    start_cancelled: AtomicBool,
    /// Set once Jaull has planned the server: from then on llama-server may be up.
    launching: AtomicBool,
}

fn read_key(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|key| key.trim().to_string()).filter(|key| !key.is_empty())
}

fn set_server_ready(window: &AppWindow, server: &Server, endpoint: Endpoint, info: ServerInfo) {
    window.set_server_url(endpoint.url.clone().into());
    window.set_server_key(endpoint.key.clone().into());
    window.set_server_lan_url(info.lan_url.into());
    window.set_firewall_rule(info.firewall_rule.into());
    window.set_server_sharing(info.sharing.into());
    *server.endpoint.lock().unwrap() = Some(endpoint);
    window.set_server_running(true);
    window.set_server_status(format!("{} is running.", info.alias).into());
}

/// At start-up: is a Jaull server already running (started by the TUI, the CLI or an earlier window)?
pub fn check_server(window: &AppWindow, server: &Arc<Server>) {
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let Ok(out) = jaull::command().args(["serve", "status", "--json"]).output() else { return };
        let Ok(status) = serde_json::from_slice::<Value>(&out.stdout) else { return };
        if status["running"].as_bool() != Some(true) || status["healthy"].as_bool() != Some(true) {
            return;
        }
        let (Some(url), Some(key)) = (status["url"].as_str().map(str::to_string), status["api_key_path"].as_str().and_then(read_key)) else { return };
        let alias = text(&status["alias"]);
        let endpoint = Endpoint { url, key, alias: alias.clone(), cert: status["tls_cert"].as_str().map(str::to_string) };
        let info = ServerInfo { alias, sharing: "Started outside this window.".into(), ..Default::default() };
        let context = status["context_per_user"].as_u64().unwrap_or(0);
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(window) = weak.upgrade() {
                if !window.get_server_busy() && !window.get_server_running() {
                    server.context_per_user.store(context, Ordering::SeqCst);
                    set_server_ready(&window, &server, endpoint, info);
                }
            }
        });
    });
}

/// How to start the server: for this window's chat, or for several people.
pub struct ServeRequest {
    pub users: i32,
    /// Tokens per person; empty leaves Jaull's default.
    pub context: String,
    pub shared: bool,
    pub lan: bool,
}

/// What `serve start` ended with.
enum Outcome {
    Started(Endpoint, ServerInfo, u64),
    /// The message, and the backend to install when llama.cpp is what is missing.
    Failed(String, Option<String>),
}

fn started(event: &Value) -> Outcome {
    let Some(key) = event["api_key_path"].as_str().and_then(read_key) else {
        return Outcome::Failed("The server started but its API key could not be read.".into(), None);
    };
    let users = event["users"].as_u64().unwrap_or(1);
    let context = event["context_per_user"].as_u64().unwrap_or(0);
    let pool = event["total_context"].as_u64().unwrap_or(context);
    let sharing = match users {
        1 => format!("1 person, {} tokens of context.", thousands(context)),
        n if context.saturating_mul(n) > pool => {
            format!("{n} people at once sharing {} tokens (up to {} per person).", thousands(pool), thousands(context))
        }
        n => format!("{n} people at once, {} tokens each.", thousands(context)),
    };
    let endpoint = Endpoint {
        url: text(&event["url"]),
        key,
        alias: text(&event["alias"]),
        cert: event["tls_cert"].as_str().map(str::to_string),
    };
    let info = ServerInfo {
        alias: text(&event["alias"]),
        lan_url: event["lan_url"].as_str().unwrap_or("").to_string(),
        firewall_rule: event["firewall_rule"].as_str().unwrap_or("").to_string(),
        sharing,
    };
    Outcome::Started(endpoint, info, context)
}

fn failed(event: &Value) -> Outcome {
    // `reason` says what went wrong without reading the message (jaull serve start --json).
    let backend = (event["reason"].as_str() == Some("runtime_missing"))
        .then(|| event["backend"].as_str().map(str::to_string))
        .flatten();
    Outcome::Failed(text(&event["message"]), backend)
}

pub fn use_model(window: &AppWindow, server: &Arc<Server>, repo: &str, quant: &str, request: ServeRequest) {
    if window.get_server_busy() || window.get_server_running() || window.get_replying() {
        return;
    }
    let mut cmd = jaull::command();
    cmd.args(["serve", "start", repo, "--json", "--no-opencode-config"]);
    if !quant.is_empty() {
        cmd.args(["-q", quant]);
    }
    if !request.context.is_empty() {
        cmd.args(["--context", &request.context]);
    }
    cmd.args(["--users", &request.users.max(1).to_string()]);
    if request.shared {
        cmd.arg("--shared-context");
    }
    if request.lan {
        cmd.arg("--lan");
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            window.set_server_status(format!("Could not start Jaull: {e}").into());
            return;
        }
    };
    *server.start_pid.lock().unwrap() = Some(child.id());
    server.start_cancelled.store(false, Ordering::SeqCst);
    server.launching.store(false, Ordering::SeqCst);
    window.set_server_busy(true);
    window.set_server_starting(true);
    window.set_runtime_backend("".into());
    window.set_page(4);
    window.set_server_status(format!("Preparing {repo}… The first time it downloads the model, which can take a while.").into());
    let stdout = child.stdout.take().expect("stdout is piped");
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let mut outcome = None;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(event) = serde_json::from_str::<Value>(&line) else { continue };
            match event["event"].as_str() {
                Some("progress") => {
                    let message = text(&event["message"]);
                    let weak = weak.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(window) = weak.upgrade() {
                            window.set_server_status(message.into());
                        }
                    });
                }
                Some("plan") => server.launching.store(true, Ordering::SeqCst),
                Some("started") => {
                    server.started_here.store(true, Ordering::SeqCst);
                    outcome = Some(started(&event));
                }
                Some("error") => outcome = Some(failed(&event)),
                _ => {}
            }
        }
        let _ = child.wait();
        *server.start_pid.lock().unwrap() = None;
        let cancelled = server.start_cancelled.load(Ordering::SeqCst);
        if cancelled {
            // Ended halfway: a llama-server it had already launched must not stay behind.
            if server.launching.load(Ordering::SeqCst) || server.started_here.load(Ordering::SeqCst) {
                jaull::serve_stop();
            }
            server.started_here.store(false, Ordering::SeqCst);
        }
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            window.set_server_busy(false);
            window.set_server_starting(false);
            if cancelled {
                window.set_server_status("Cancelled. Nothing is running.".into());
                return;
            }
            match outcome {
                Some(Outcome::Started(endpoint, info, context)) => {
                    server.history.lock().unwrap().clear();
                    server.context_per_user.store(context, Ordering::SeqCst);
                    window.set_messages(ModelRc::from(Rc::new(VecModel::<Msg>::default())));
                    window.set_chat_note("".into());
                    set_server_ready(&window, &server, endpoint, info);
                }
                Some(Outcome::Failed(message, backend)) => {
                    if let Some(backend) = backend {
                        window.set_runtime_backend(backend.into());
                    }
                    window.set_server_status(format!("Could not start the model: {message}").into());
                }
                None => window.set_server_status("Jaull ended without starting the model.".into()),
            }
        });
    });
}

/// Cancel: ends the `serve start` that is downloading or starting the model.
pub fn cancel_start(window: &AppWindow, server: &Server) {
    let Some(pid) = *server.start_pid.lock().unwrap() else { return };
    server.start_cancelled.store(true, Ordering::SeqCst);
    window.set_server_status("Cancelling…".into());
    jaull::kill_tree(pid);
}

pub fn stop_model(window: &AppWindow, server: &Arc<Server>) {
    window.set_server_busy(true);
    window.set_server_status("Stopping the model…".into());
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let ok = jaull::serve_stop();
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            window.set_server_busy(false);
            if ok {
                *server.endpoint.lock().unwrap() = None;
                server.started_here.store(false, Ordering::SeqCst);
                window.set_server_running(false);
                window.set_server_status("The model is stopped and its memory is free.".into());
            } else {
                window.set_server_status("Jaull could not stop the model.".into());
            }
        });
    });
}

/// When the window closes: end a start that is still running, and stop a model this
/// window started, so neither keeps gigabytes of memory busy. Other servers are left alone.
pub fn shut_down(server: &Server) {
    let starting = *server.start_pid.lock().unwrap();
    if let Some(pid) = starting {
        server.start_cancelled.store(true, Ordering::SeqCst);
        jaull::kill_tree(pid);
    }
    let launched = starting.is_some() && server.launching.load(Ordering::SeqCst);
    if launched || server.started_here.load(Ordering::SeqCst) {
        jaull::serve_stop();
    }
}

/// `jaull runtime install --backend <b>`: the pinned llama.cpp build, downloaded and checked.
pub fn install_runtime(window: &AppWindow, backend: &str) {
    window.set_runtime_installing(true);
    window.set_server_status(format!("Installing llama.cpp ({backend})… a few hundred MB.").into());
    let mut cmd = jaull::command();
    cmd.args(["runtime", "install", "--backend", backend]).env("NO_COLOR", "1");
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let result = cmd.output();
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            window.set_runtime_installing(false);
            match result {
                Ok(out) if out.status.success() => {
                    window.set_runtime_backend("".into());
                    window.set_server_status("llama.cpp is installed. Start the model again.".into());
                }
                Ok(out) => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    window.set_server_status(format!("Could not install llama.cpp: {}", jaull::strip_ansi(err.trim())).into());
                }
                Err(e) => window.set_server_status(format!("Could not start Jaull: {e}").into()),
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::{failed, Outcome};
    use serde_json::json;

    #[test]
    fn a_missing_llama_cpp_offers_the_backend_jaull_names() {
        let event = json!({"event": "error", "code": 6, "reason": "runtime_missing", "backend": "vulkan",
                           "message": "llama-server was not found."});
        let Outcome::Failed(message, backend) = failed(&event) else { panic!("expected a failure") };
        assert_eq!((message.as_str(), backend.as_deref()), ("llama-server was not found.", Some("vulkan")));
        // Other errors offer nothing, even if their text mentions an install command.
        let other = json!({"event": "error", "code": 5, "message": "Run `jaull runtime install --backend cpu`"});
        let Outcome::Failed(_, backend) = failed(&other) else { panic!("expected a failure") };
        assert_eq!(backend, None);
    }
}
