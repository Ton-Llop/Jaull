// Jaull desktop, experiment: a light window over the existing Python engine.
// The engine runs only while it is needed (one `python -m jaull ... --json` process per
// action), so nothing but this window stays in memory when the app is idle.
#![windows_subsystem = "windows"]

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use slint::{Model, ModelRc, SharedString, VecModel};

slint::include_modules!();

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const USE_CASES: [&str; 6] = [
    "general_chat",
    "coding",
    "document_qa",
    "summarization_extraction",
    "reasoning",
    "writing_translation",
];
const PRIORITIES: [&str; 4] = ["balanced", "quality", "speed", "memory"];

/// How to start Jaull: JAULL_PYTHON (an interpreter with Jaull installed), then the Python
/// the installer puts next to this executable, then this checkout's virtualenv, then a
/// `jaull` on PATH. Python is started directly, with `-m jaull`, so that cancelling a
/// search can end the whole process tree.
fn jaull_command() -> Command {
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
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Ends a process and its children (the virtualenv's python.exe starts another one).
fn kill_tree(pid: u32) {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/F", "/T", "/PID", &pid.to_string()]).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let _ = cmd.status();
}

fn run_jaull_json(args: &[&str]) -> Result<Value, String> {
    let out = jaull_command()
        .args(args)
        .output()
        .map_err(|e| format!("Could not start Jaull: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("Jaull failed: {}", err.trim()));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("Unexpected output from Jaull: {e}"))
}

fn gib(value: &Value) -> Option<String> {
    value.as_f64().map(|bytes| format!("{:.1} GiB", bytes / 1_073_741_824.0))
}

fn text(value: &Value) -> String {
    value.as_str().unwrap_or("?").to_string()
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Turn the `hardware` object of `jaull scan --json` into label/value rows.
fn hardware_rows(hw: &Value) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    let os = match hw["os_version"].as_str() {
        Some(version) => format!("{} ({version})", text(&hw["os"])),
        None => text(&hw["os"]),
    };
    rows.push(("Operating system".into(), os));
    rows.push(("Processor".into(), text(&hw["cpu"]["model"])));
    rows.push((
        "Cores".into(),
        format!(
            "{} physical, {} logical",
            hw["cpu"]["physical_cores"].as_u64().map_or("?".into(), |n| n.to_string()),
            hw["cpu"]["logical_cores"].as_u64().map_or("?".into(), |n| n.to_string())
        ),
    ));
    rows.push((
        "Memory (RAM)".into(),
        format!(
            "{} total, {} free now",
            gib(&hw["memory"]["total_bytes"]).unwrap_or_else(|| "?".into()),
            gib(&hw["memory"]["available_bytes"]).unwrap_or_else(|| "?".into())
        ),
    ));
    for gpu in hw["gpus"].as_array().into_iter().flatten() {
        let vram = gib(&gpu["vram_total_bytes"]).map_or(String::new(), |v| format!(", {v} of video memory"));
        rows.push(("Graphics card".into(), format!("{}{vram}", text(&gpu["name"]))));
    }
    for acc in hw["accelerators"].as_array().into_iter().flatten() {
        let memory = match gib(&acc["dedicated_memory_bytes"]) {
            Some(v) => format!("{v} dedicated"),
            None if acc["shared_memory"].as_bool() == Some(true) => "shares the system RAM".into(),
            None => "memory unknown".into(),
        };
        let backends: Vec<String> = acc["backends"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|b| b["availability"].as_str() == Some("available"))
            .map(|b| text(&b["backend"]).to_uppercase())
            .collect();
        let usable = if backends.is_empty() { "none detected".to_string() } else { backends.join(", ") };
        rows.push((
            "Accelerator".into(),
            format!("{} ({}, {memory}); can run with: {usable}", text(&acc["name"]), text(&acc["type"])),
        ));
    }
    for disk in hw["storage"].as_array().into_iter().flatten() {
        rows.push((
            format!("Disk {}", text(&disk["mountpoint"])),
            format!(
                "{} free of {}",
                gib(&disk["available_bytes"]).unwrap_or_else(|| "?".into()),
                gib(&disk["total_bytes"]).unwrap_or_else(|| "?".into())
            ),
        ));
    }
    for warning in hw["warnings"].as_array().into_iter().flatten() {
        rows.push(("Warning".into(), text(warning)));
    }
    rows
}

/// One line for the header: cores, RAM and the first accelerator with a usable backend.
fn machine_summary(hw: &Value) -> String {
    let mut parts = Vec::new();
    if let (Some(p), Some(l)) = (hw["cpu"]["physical_cores"].as_u64(), hw["cpu"]["logical_cores"].as_u64()) {
        parts.push(format!("{p}C/{l}T"));
    }
    if let Some(ram) = gib(&hw["memory"]["total_bytes"]) {
        parts.push(format!("{ram} RAM"));
    }
    for acc in hw["accelerators"].as_array().into_iter().flatten() {
        let backend = acc["backends"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|b| b["availability"].as_str() == Some("available"))
            .map(|b| text(&b["backend"]));
        parts.push(match backend {
            Some(b) => format!("{} · {b}", text(&acc["name"])),
            None => text(&acc["name"]),
        });
        break;
    }
    parts.join(" · ")
}

/// Plain words for Jaull's compatibility status.
fn fit_words(status: &str) -> &'static str {
    match status {
        "comfortable" => "Fits comfortably",
        "compatible" => "Fits",
        "tight" => "Fits, with little room to spare",
        "memory_pressure" => "Fits only if other programs are closed",
        "offloading_required" => "Runs partly on the processor, slower",
        "insufficient" => "Does not fit in this computer",
        _ => "Fit not known",
    }
}

/// One card per recommendation in the report of `jaull recommend --json`.
fn recommendation_cards(report: &Value) -> Vec<Rec> {
    report["recommendations"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|rec| {
            let quant = rec["artifact"]["quantization"].as_str().unwrap_or("");
            let fit = if quant.is_empty() {
                fit_words(rec["compatibility"].as_str().unwrap_or("")).to_string()
            } else {
                format!("{} · {quant}", fit_words(rec["compatibility"].as_str().unwrap_or("")))
            };
            let needs = gib(&rec["memory"]["memory"]["total_bytes"]);
            let context = rec["memory"]["inference_configuration"]["context_length"].as_u64();
            let detail = match (needs, context) {
                (Some(n), Some(c)) => format!("Needs about {n} with a {}-token context", thousands(c)),
                (Some(n), None) => format!("Needs about {n}"),
                _ => "Memory not estimated".to_string(),
            };
            let confidence = rec["confidence"].as_str().unwrap_or("unknown");
            let total_bytes = rec["memory"]["memory"]["total_bytes"].as_f64();
            let lines = |key: &str| -> String {
                rec[key].as_array().into_iter().flatten().map(|r| format!("· {}", text(r))).collect::<Vec<_>>().join("\n")
            };
            let runtime = rec["runtime_assessment"]["runtime"].as_str().unwrap_or("llama.cpp");
            let format = rec["artifact"]["format"].as_str().unwrap_or("gguf").to_uppercase();
            let _ = fit;
            Rec {
                repo: SharedString::from(text(&rec["repo_id"])),
                quant: SharedString::from(quant),
                // The server starts with the context this card's memory was estimated for.
                context: SharedString::from(context.map(|c| c.to_string()).unwrap_or_default()),
                rank: SharedString::from(rec["rank"].as_u64().map_or("?".into(), |r| r.to_string())),
                name: SharedString::from(text(&rec["repo_id"])),
                fit: SharedString::from(fit_words(rec["compatibility"].as_str().unwrap_or(""))),
                fit_key: SharedString::from(rec["compatibility"].as_str().unwrap_or("unknown")),
                total: SharedString::from(total_bytes.map_or("memory unknown".into(), |b| format!("{:.2} GiB", b / 1_073_741_824.0))),
                total_gib: total_bytes.map_or(0.0, |b| (b / 1_073_741_824.0) as f32),
                stack: SharedString::from(format!("{runtime} · {format} · {quant}")),
                detail: SharedString::from(format!("{detail} · estimate confidence: {confidence}")),
                reason: SharedString::from(
                    rec["reasons"].as_array().and_then(|r| r.first()).map(text).unwrap_or_default(),
                ),
                reasons: SharedString::from(lines("reasons")),
                warnings: SharedString::from(lines("warnings")),
            }
        })
        .collect()
}

fn start_scan(window: &AppWindow) {
    window.set_busy(true);
    window.set_status("Detecting hardware… (a few seconds)".into());
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let result = run_jaull_json(&["scan", "--json"]).map(|v| {
            let hw = &v["hardware"];
            (hardware_rows(hw), machine_summary(hw), hw["memory"]["total_bytes"].as_f64().unwrap_or(0.0))
        });
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            match result {
                Ok((rows, machine, ram_bytes)) => {
                    window.set_machine(machine.into());
                    if ram_bytes > 0.0 {
                        window.set_ram_total_gib((ram_bytes / 1_073_741_824.0) as f32);
                    }
                    let model: Vec<Row> = rows
                        .into_iter()
                        .map(|(label, value)| Row { label: SharedString::from(label), value: SharedString::from(value) })
                        .collect();
                    window.set_rows(ModelRc::from(Rc::new(VecModel::from(model))));
                    window.set_status("Detected by Jaull.".into());
                }
                Err(message) => window.set_status(message.into()),
            }
            window.set_busy(false);
        });
    });
}

/// The search that is running, so Cancel can end it.
#[derive(Default)]
struct Search {
    pid: Mutex<Option<u32>>,
    cancelled: AtomicBool,
}

fn start_search(window: &AppWindow, search: &Arc<Search>, use_case: usize, priority: usize, languages: &str, commercial: bool) {
    let mut cmd = jaull_command();
    cmd.args(["recommend", "--json", "--use-case", USE_CASES[use_case.min(USE_CASES.len() - 1)]]);
    cmd.args(["--priority", PRIORITIES[priority.min(PRIORITIES.len() - 1)]]);
    cmd.args(["--commercial-use", if commercial { "yes" } else { "no_preference" }]);
    for lang in languages.split([',', ' ', ';']).map(str::trim).filter(|l| !l.is_empty()) {
        cmd.args(["--language", &lang.to_lowercase()]);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            window.set_search_status(format!("Could not start Jaull: {e}").into());
            return;
        }
    };
    window.set_searching(true);
    window.set_steps(ModelRc::from(Rc::new(VecModel::<Step>::default())));
    window.set_recs(ModelRc::from(Rc::new(VecModel::<Rec>::default())));
    window.set_selected(0);
    window.set_search_status("Checking this machine and preparing the search (about 15 s)…".into());
    *search.pid.lock().unwrap() = Some(child.id());
    search.cancelled.store(false, Ordering::SeqCst);

    let stdout = child.stdout.take().expect("stdout is piped");
    let weak = window.as_weak();
    let search = Arc::clone(search);
    std::thread::spawn(move || {
        let mut outcome: Option<Result<Value, String>> = None;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(event) = serde_json::from_str::<Value>(&line) else { continue };
            match event["event"].as_str() {
                Some("progress") => {
                    let status = format!("Current: {}", text(&event["label"]));
                    let steps: Vec<Step> = event["steps"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|s| Step {
                            label: text(&s["label"]).into(),
                            detail: s["detail"].as_str().unwrap_or("").into(),
                            state: match s["status"].as_str() {
                                Some("done") | Some("skipped") => 2,
                                Some("running") => 1,
                                _ => 0,
                            },
                        })
                        .collect();
                    let weak = weak.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(window) = weak.upgrade() {
                            window.set_search_status(status.into());
                            window.set_steps(ModelRc::from(Rc::new(VecModel::from(steps))));
                        }
                    });
                }
                Some("result") => outcome = Some(Ok(event["report"].clone())),
                Some("error") => outcome = Some(Err(text(&event["message"]))),
                _ => {}
            }
        }
        let _ = child.wait();
        *search.pid.lock().unwrap() = None;
        let cancelled = search.cancelled.load(Ordering::SeqCst);
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            window.set_searching(false);
            if cancelled {
                window.set_search_status("Search cancelled.".into());
                return;
            }
            match outcome {
                Some(Ok(report)) => {
                    let cards = recommendation_cards(&report);
                    if cards.is_empty() {
                        let why: Vec<String> = report["no_results_reason"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(text)
                            .collect();
                        let why = if why.is_empty() { "No model fits these needs.".to_string() } else { why.join(" ") };
                        window.set_search_status(why.into());
                    } else {
                        window.set_search_status(
                            format!("{} models fit this computer, best first.", cards.len()).into(),
                        );
                    }
                    window.set_selected(0);
                    window.set_recs(ModelRc::from(Rc::new(VecModel::from(cards))));
                }
                Some(Err(message)) => window.set_search_status(format!("The search failed: {message}").into()),
                None => window.set_search_status("The search ended without a result.".into()),
            }
        });
    });
}

/// Where the chat sends its requests.
#[derive(Clone)]
struct Endpoint {
    url: String,
    key: String,
    alias: String,
    /// Jaull's self-signed certificate when the server uses HTTPS (shared on the network).
    cert: Option<String>,
}

/// What the Chat page shows about the running server.
#[derive(Default)]
struct ServerInfo {
    alias: String,
    lan_url: String,
    firewall_rule: String,
    sharing: String,
}

/// The model server Jaull runs (llama-server), as far as this window knows it.
#[derive(Default)]
struct Server {
    endpoint: Mutex<Option<Endpoint>>,
    /// Stop it when the window closes: only if this window started it.
    started_here: AtomicBool,
    /// The conversation sent with every message (role, content).
    history: Mutex<Vec<(String, String)>>,
}

fn read_key(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|key| key.trim().to_string()).filter(|key| !key.is_empty())
}

/// An HTTP client for the endpoint. With HTTPS it trusts only the certificate Jaull made,
/// never the system's store, so it talks to this server and nothing else.
fn agent_for(endpoint: &Endpoint) -> Result<ureq::Agent, String> {
    let builder = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(300))
        .redirects(0);
    let Some(cert) = &endpoint.cert else {
        if endpoint.url.starts_with("https://") {
            return Err("The running HTTPS server has no certificate path. Restart it from Jaull.".into());
        }
        return Ok(builder.build());
    };
    let pem = std::fs::read(cert).map_err(|e| format!("Could not read the server certificate: {e}"))?;
    let mut roots = rustls::RootCertStore::empty();
    for der in rustls_pemfile::certs(&mut pem.as_slice()) {
        let der = der.map_err(|e| format!("The server certificate is not valid: {e}"))?;
        roots.add(der).map_err(|e| format!("The server certificate is not valid: {e}"))?;
    }
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(builder.tls_config(Arc::new(config)).build())
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
fn check_server(window: &AppWindow, server: &Arc<Server>) {
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let Ok(out) = jaull_command().args(["serve", "status", "--json"]).output() else { return };
        let Ok(status) = serde_json::from_slice::<Value>(&out.stdout) else { return };
        if status["running"].as_bool() != Some(true) || status["healthy"].as_bool() != Some(true) {
            return;
        }
        let (Some(url), Some(key)) = (status["url"].as_str().map(str::to_string), status["api_key_path"].as_str().and_then(read_key)) else { return };
        let alias = text(&status["alias"]);
        let endpoint = Endpoint { url, key, alias: alias.clone(), cert: status["tls_cert"].as_str().map(str::to_string) };
        let info = ServerInfo { alias, sharing: "Started outside this window.".into(), ..Default::default() };
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(window) = weak.upgrade() {
                if !window.get_server_busy() && !window.get_server_running() {
                    set_server_ready(&window, &server, endpoint, info);
                }
            }
        });
    });
}

/// How to start the server: for this window's chat, or for several people.
struct ServeRequest {
    users: i32,
    /// Tokens per person; empty leaves Jaull's default.
    context: String,
    shared: bool,
    lan: bool,
}

fn use_model(window: &AppWindow, server: &Arc<Server>, repo: &str, quant: &str, request: ServeRequest) {
    if window.get_server_busy() || window.get_server_running() || window.get_replying() {
        return;
    }
    let mut cmd = jaull_command();
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
    window.set_server_busy(true);
    window.set_page(4);
    window.set_server_status(format!("Preparing {repo}… The first time it downloads the model, which can take a while.").into());
    let stdout = child.stdout.take().expect("stdout is piped");
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let mut outcome: Option<Result<(Endpoint, ServerInfo), String>> = None;
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
                Some("started") => {
                    server.started_here.store(true, Ordering::SeqCst);
                    let key = event["api_key_path"].as_str().and_then(read_key);
                    outcome = Some(match key {
                        Some(key) => {
                            let users = event["users"].as_u64().unwrap_or(1);
                            let context = event["context_per_user"].as_u64().unwrap_or(0);
                            let pool = event["total_context"].as_u64().unwrap_or(context);
                            let sharing = match users {
                                1 => format!("1 person, {} tokens of context.", thousands(context)),
                                n if context.saturating_mul(n) > pool => format!(
                                    "{n} people at once sharing {} tokens (up to {} per person).",
                                    thousands(pool), thousands(context)
                                ),
                                n => format!("{n} people at once, {} tokens each.", thousands(context)),
                            };
                            Ok((
                                Endpoint {
                                    url: text(&event["url"]),
                                    key,
                                    alias: text(&event["alias"]),
                                    cert: event["tls_cert"].as_str().map(str::to_string),
                                },
                                ServerInfo {
                                    alias: text(&event["alias"]),
                                    lan_url: event["lan_url"].as_str().unwrap_or("").to_string(),
                                    firewall_rule: event["firewall_rule"].as_str().unwrap_or("").to_string(),
                                    sharing,
                                },
                            ))
                        }
                        None => Err("The server started but its API key could not be read.".into()),
                    });
                }
                Some("error") => outcome = Some(Err(text(&event["message"]))),
                _ => {}
            }
        }
        let _ = child.wait();
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            window.set_server_busy(false);
            match outcome {
                Some(Ok((endpoint, info))) => {
                    server.history.lock().unwrap().clear();
                    window.set_messages(ModelRc::from(Rc::new(VecModel::<Msg>::default())));
                    set_server_ready(&window, &server, endpoint, info);
                }
                Some(Err(message)) => {
                    // "llama-server was not found. Install it with `jaull runtime install --backend vulkan`."
                    if let Some(rest) = message.split("runtime install --backend ").nth(1) {
                        let backend: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
                        window.set_runtime_backend(backend.into());
                    }
                    window.set_server_status(format!("Could not start the model: {message}").into());
                }
                None => window.set_server_status("Jaull ended without starting the model.".into()),
            }
        });
    });
}

fn stop_model(window: &AppWindow, server: &Arc<Server>) {
    window.set_server_busy(true);
    window.set_server_status("Stopping the model…".into());
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let ok = jaull_command().args(["serve", "stop"]).stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
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

fn messages_model(window: &AppWindow) -> Rc<VecModel<Msg>> {
    let current = window.get_messages();
    if let Some(model) = current.as_any().downcast_ref::<VecModel<Msg>>() {
        // Rebuild an Rc around the same rows: Slint keeps the ModelRc we set below.
        let rows: Vec<Msg> = (0..model.row_count()).filter_map(|i| model.row_data(i)).collect();
        let model = Rc::new(VecModel::from(rows));
        window.set_messages(ModelRc::from(model.clone()));
        return model;
    }
    let model = Rc::new(VecModel::<Msg>::default());
    window.set_messages(ModelRc::from(model.clone()));
    model
}

fn read_chat_stream(reader: impl BufRead, mut on_text: impl FnMut(&str)) -> Result<(), String> {
    for line in reader.lines() {
        let line = line.map_err(|e| format!("The model response was interrupted: {e}"))?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(());
        }
        let chunk: Value = serde_json::from_str(data)
            .map_err(|_| "The server sent an invalid streaming response.".to_string())?;
        if !chunk["error"].is_null() {
            return Err("The server reported an error during generation.".into());
        }
        if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str() {
            on_text(text);
        }
    }
    Err("The model response ended before it was complete.".into())
}

#[cfg(test)]
mod chat_tests {
    use super::read_chat_stream;

    const CHUNK: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\r\n\r\n";

    #[test]
    fn complete_stream_emits_text() {
        let event_stream = format!("{CHUNK}data: [DONE]\r\n\r\n");
        let mut answer = String::new();
        read_chat_stream(event_stream.as_bytes(), |piece| answer.push_str(piece)).unwrap();
        assert_eq!(answer, "hello");
    }

    #[test]
    fn incomplete_and_error_streams_fail() {
        assert!(read_chat_stream(CHUNK.as_bytes(), |_| {}).is_err());
        let error_stream = b"data: {\"error\":{\"message\":\"generation failed\"}}\n\n";
        assert!(read_chat_stream(error_stream.as_slice(), |_| {}).is_err());
    }
}

/// Sends the conversation to llama-server and streams the answer into the last bubble.
fn send(window: &AppWindow, server: &Arc<Server>, message: &str) {
    if window.get_server_busy() || window.get_replying() || message.trim().is_empty() {
        return;
    }
    let Some(endpoint) = server.endpoint.lock().unwrap().clone() else { return };
    let model = messages_model(window);
    model.push(Msg { mine: true, text: SharedString::from(message) });
    model.push(Msg { mine: false, text: SharedString::from("…") });
    window.set_replying(true);

    let mut history = server.history.lock().unwrap().clone();
    history.push(("user".into(), message.to_string()));
    let weak = window.as_weak();
    let server = Arc::clone(server);
    std::thread::spawn(move || {
        let messages: Vec<Value> = history.iter().map(|(role, content)| serde_json::json!({"role": role, "content": content})).collect();
        let body = serde_json::json!({"model": endpoint.alias, "messages": messages, "stream": true});
        let mut answer = String::new();
        let request = agent_for(&endpoint).map(|agent| {
            agent
                .post(&format!("{}/v1/chat/completions", endpoint.url))
                .set("Authorization", &format!("Bearer {}", endpoint.key))
                .send_json(body)
        });
        let error = match request {
            Err(message) => Some(message),
            Ok(sent) => match sent {
            Ok(response) => {
                read_chat_stream(BufReader::new(response.into_reader()), |piece| {
                        answer.push_str(piece);
                        let shown = answer.clone();
                        let weak = weak.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(window) = weak.upgrade() {
                                let rows = window.get_messages();
                                let last = rows.row_count().saturating_sub(1);
                                rows.set_row_data(last, Msg { mine: false, text: shown.into() });
                            }
                        });
                }).err()
            }
            Err(ureq::Error::Status(code, _)) => Some(format!("The server answered with error {code}.")),
            Err(e) => Some(format!("Could not reach the model: {e}")),
            },
        };
        if error.is_none() {
            let mut kept = server.history.lock().unwrap();
            kept.push(("user".into(), history.last().map(|(_, c)| c.clone()).unwrap_or_default()));
            kept.push(("assistant".into(), answer.clone()));
        }
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            if let Some(error) = error {
                let rows = window.get_messages();
                let last = rows.row_count().saturating_sub(1);
                rows.set_row_data(last, Msg { mine: false, text: error.into() });
            }
            window.set_replying(false);
        });
    });
}

/// Reads the downloaded files (`jaull storage list --json`) for Set up a local server and Free disk space.
fn refresh_storage(window: &AppWindow) {
    window.set_storage_busy(true);
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let result = run_jaull_json(&["storage", "list", "--json"]);
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            window.set_storage_busy(false);
            let listing = match result {
                Ok(listing) => listing,
                Err(message) => {
                    window.set_storage_status(message.into());
                    return;
                }
            };
            let mut total = 0.0;
            let models: Vec<LocalModel> = listing["models"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|m| {
                    total += m["size_bytes"].as_f64().unwrap_or(0.0);
                    let note = match (m["verified"].as_bool(), m["measured"].as_bool(), m["quantization"].as_str()) {
                        (_, _, None) => "quantization not recognised: start it from the CLI",
                        (Some(false), _, _) => "not verified",
                        (_, Some(true), _) => "quality measured on this file",
                        _ => "",
                    };
                    LocalModel {
                        repo: text(&m["repo_id"]).into(),
                        filename: text(&m["filename"]).into(),
                        quant: m["quantization"].as_str().unwrap_or("").into(),
                        size: gib(&m["size_bytes"]).unwrap_or_else(|| "?".into()).into(),
                        note: note.into(),
                    }
                })
                .collect();
            let logs = &listing["evaluation_logs"];
            window.set_logs_size(gib(&logs["size_bytes"]).unwrap_or_else(|| "?".into()).into());
            window.set_storage_status(
                format!(
                    "{} model files, {:.1} GiB in total. Evaluation logs: {} folders.",
                    models.len(),
                    total / 1_073_741_824.0,
                    logs["folders"].as_u64().unwrap_or(0)
                )
                .into(),
            );
            let first_usable = models.iter().position(|m| !m.quant.is_empty()).unwrap_or(0);
            window.set_setup_selected(first_usable as i32);
            window.set_local_models(ModelRc::from(Rc::new(VecModel::from(models))));
        });
    });
}

/// `jaull storage remove`: one model file, or every evaluation log folder.
fn remove_storage(window: &AppWindow, args: Vec<String>) {
    window.set_storage_busy(true);
    window.set_storage_status("Removing…".into());
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let mut all = vec!["storage".to_string(), "remove".to_string()];
        all.extend(args);
        all.push("--json".into());
        let refs: Vec<&str> = all.iter().map(String::as_str).collect();
        let result = run_jaull_json(&refs);
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            refresh_storage(&window);
            match result {
                Ok(v) if v["error"].is_null() => {
                    window.set_storage_status(format!("Freed {}.", gib(&v["freed_bytes"]).unwrap_or_else(|| "?".into())).into())
                }
                Ok(v) => window.set_storage_status(text(&v["error"]).into()),
                // A file the running server has open cannot be deleted on Windows.
                Err(message) => window.set_storage_status(format!("{message} Stop the model first if it is using this file.").into()),
            }
        });
    });
}

/// `jaull runtime install --backend <b>`: the pinned llama.cpp build, downloaded and checked.
fn install_runtime(window: &AppWindow, backend: &str) {
    window.set_runtime_installing(true);
    window.set_server_status(format!("Installing llama.cpp ({backend})… a few hundred MB.").into());
    let mut cmd = jaull_command();
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
                    window.set_server_status(format!("Could not install llama.cpp: {}", strip_ansi(err.trim())).into());
                }
                Err(e) => window.set_server_status(format!("Could not start Jaull: {e}").into()),
            }
        });
    });
}

/// Removes terminal colour codes, in case a tool prints them anyway.
fn strip_ansi(text: &str) -> String {
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

/// Advanced tools: runs `jaull inspect|estimate|doctor` and shows its plain-text report.
fn run_tool(window: &AppWindow, tool: i32, repo: &str, quant: &str, context: &str) {
    let mut cmd = jaull_command();
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
                    text.push_str("\n");
                    text.push_str(&err);
                }
                strip_ansi(&text)
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

fn main() -> Result<(), slint::PlatformError> {
    // The software renderer keeps the window at a few MiB; the default OpenGL one loaded
    // the graphics driver and used about 110 MiB in the hello-world measurement.
    if std::env::var_os("SLINT_BACKEND").is_none() {
        std::env::set_var("SLINT_BACKEND", "winit-software");
    }
    let window = AppWindow::new()?;
    let weak = window.as_weak();
    window.window().on_close_requested(move || {
        if let Some(window) = weak.upgrade() {
            if window.get_server_busy() {
                window.set_page(4);
                window.set_server_status("Wait until the server operation finishes before closing.".into());
                return slint::CloseRequestResponse::KeepWindowShown;
            }
        }
        slint::CloseRequestResponse::HideWindow
    });
    let search = Arc::new(Search::default());

    let weak = window.as_weak();
    window.on_rescan(move || {
        if let Some(window) = weak.upgrade() {
            start_scan(&window);
        }
    });

    let weak = window.as_weak();
    let for_find = Arc::clone(&search);
    window.on_find(move |use_case, priority, languages, commercial| {
        if let Some(window) = weak.upgrade() {
            start_search(&window, &for_find, use_case.max(0) as usize, priority.max(0) as usize, &languages, commercial);
        }
    });

    let weak = window.as_weak();
    let for_cancel = Arc::clone(&search);
    window.on_cancel(move || {
        if let Some(pid) = *for_cancel.pid.lock().unwrap() {
            for_cancel.cancelled.store(true, Ordering::SeqCst);
            kill_tree(pid);
            if let Some(window) = weak.upgrade() {
                window.set_search_status("Cancelling…".into());
            }
        }
    });

    let server = Arc::new(Server::default());
    let weak = window.as_weak();
    let for_use = Arc::clone(&server);
    window.on_use_model(move |repo, quant, context| {
        if let Some(window) = weak.upgrade() {
            let request = ServeRequest { users: 1, context: context.to_string(), shared: false, lan: false };
            use_model(&window, &for_use, &repo, &quant, request);
        }
    });
    let weak = window.as_weak();
    let for_serve = Arc::clone(&server);
    window.on_serve_model(move |repo, quant, users, context, shared, lan| {
        if let Some(window) = weak.upgrade() {
            let request = ServeRequest { users, context: context.to_string(), shared, lan };
            use_model(&window, &for_serve, &repo, &quant, request);
        }
    });
    let weak = window.as_weak();
    let for_stop = Arc::clone(&server);
    window.on_stop_model(move || {
        if let Some(window) = weak.upgrade() {
            stop_model(&window, &for_stop);
        }
    });
    let weak = window.as_weak();
    let for_send = Arc::clone(&server);
    window.on_send(move |message| {
        if let Some(window) = weak.upgrade() {
            send(&window, &for_send, &message);
        }
    });

    let weak = window.as_weak();
    window.on_refresh_storage(move || {
        if let Some(window) = weak.upgrade() {
            refresh_storage(&window);
        }
    });
    let weak = window.as_weak();
    window.on_remove_model(move |repo, filename| {
        if let Some(window) = weak.upgrade() {
            remove_storage(&window, vec![repo.to_string(), filename.to_string()]);
        }
    });
    let weak = window.as_weak();
    window.on_remove_logs(move || {
        if let Some(window) = weak.upgrade() {
            remove_storage(&window, vec!["--evaluation-logs".to_string()]);
        }
    });
    let weak = window.as_weak();
    window.on_install_runtime(move |backend| {
        if let Some(window) = weak.upgrade() {
            install_runtime(&window, &backend);
        }
    });
    let weak = window.as_weak();
    window.on_run_tool(move |tool, repo, quant, context| {
        if let Some(window) = weak.upgrade() {
            run_tool(&window, tool, &repo, &quant, &context);
        }
    });

    start_scan(&window);
    check_server(&window, &server);
    let result = window.run();
    // Closing the window must not leave a search running in the background, nor a model
    // this window started holding gigabytes of memory.
    if let Some(pid) = *search.pid.lock().unwrap() {
        kill_tree(pid);
    }
    if server.started_here.load(Ordering::SeqCst) {
        let _ = jaull_command().args(["serve", "stop"]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    result
}
