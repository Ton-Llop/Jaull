//! Your needs and Results: `jaull recommend --json` with the TUI wizard's answers.

use std::io::{BufRead, BufReader};
use std::process::Stdio;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};

use crate::format::{gib, text, thousands, GIB};
use crate::{jaull, AppWindow, Rec, Step};

// The CLI values for each option, in the order the window lists them (the TUI's order).
const USE_CASES: [&str; 6] =
    ["general_chat", "coding", "document_qa", "summarization_extraction", "reasoning", "writing_translation"];
const WORKLOAD_MODES: [&str; 2] = ["interactive", "batch"];
const PRIORITIES: [&str; 4] = ["quality", "balanced", "speed", "memory"];
const CONCURRENCY: [&str; 4] = ["single", "small", "medium", "large"];
const DOCUMENT_SCALES: [&str; 4] = ["short", "medium", "long", "collection"];
const COMMERCIAL: [&str; 4] = ["yes", "no", "no_preference", "not_sure"];
const DOCUMENT_QA: i32 = 2;

/// The search that is running, so Cancel can end it.
#[derive(Default)]
pub struct Search {
    pub pid: Mutex<Option<u32>>,
    cancelled: AtomicBool,
}

impl Search {
    pub fn cancel(&self) -> bool {
        let Some(pid) = *self.pid.lock().unwrap() else { return false };
        self.cancelled.store(true, Ordering::SeqCst);
        jaull::kill_tree(pid);
        true
    }
}

fn pick(options: &[&'static str], index: i32) -> &'static str {
    options[(index.max(0) as usize).min(options.len() - 1)]
}

/// The `jaull recommend` flags for what the Your needs screen shows.
fn answer_args(window: &AppWindow) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    let mut flag = |name: &str, value: &str| {
        args.push(name.to_string());
        args.push(value.to_string());
    };
    let use_case = window.get_use_case();
    flag("--use-case", pick(&USE_CASES, use_case));
    flag("--mode", pick(&WORKLOAD_MODES, window.get_workload_mode()));
    flag("--priority", pick(&PRIORITIES, window.get_priority()));
    flag("--concurrency", pick(&CONCURRENCY, window.get_concurrency()));
    if use_case == DOCUMENT_QA {
        flag("--document-scale", pick(&DOCUMENT_SCALES, window.get_doc_scale()));
    }
    flag("--commercial-use", pick(&COMMERCIAL, window.get_commercial()));
    let listed = [
        (window.get_lang_es(), "Spanish"),
        (window.get_lang_ca(), "Catalan"),
        (window.get_lang_en(), "English"),
        (window.get_lang_pt(), "Portuguese"),
    ];
    for (_, name) in listed.iter().filter(|(on, _)| *on) {
        flag("--language", name);
    }
    if window.get_lang_other() {
        // Jaull checks these with the wizard's own normalisation and refuses unknown ones.
        for other in window.get_other_languages().split(',').map(str::trim).filter(|l| !l.is_empty()) {
            flag("--other-language", other);
        }
    }
    args
}

pub fn start_search(window: &AppWindow, search: &Arc<Search>) {
    let mut cmd = jaull::command();
    cmd.args(["recommend", "--json"]).args(answer_args(window));
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

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
    let stderr = child.stderr.take().expect("stderr is piped");
    // Read stderr apart so a chatty library cannot fill the pipe and stall the search.
    let errors = std::thread::spawn(move || {
        let mut last = String::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if !line.trim().is_empty() {
                last = line;
            }
        }
        last
    });
    let weak = window.as_weak();
    let ram_total = window.get_ram_total_gib();
    let search = Arc::clone(search);
    std::thread::spawn(move || {
        let mut outcome: Option<Result<Value, String>> = None;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(event) = serde_json::from_str::<Value>(&line) else { continue };
            match event["event"].as_str() {
                Some("progress") => {
                    let status = format!("Current: {}", text(&event["label"]));
                    let steps = progress_steps(&event);
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
        let last_error = errors.join().unwrap_or_default();
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
                Some(Ok(report)) => show_report(&window, &report, ram_total),
                Some(Err(message)) => window.set_search_status(format!("The search failed: {message}").into()),
                // e.g. a language Jaull does not recognise: the CLI says so on stderr.
                None if !last_error.is_empty() => window.set_search_status(last_error.into()),
                None => window.set_search_status("The search ended without a result.".into()),
            }
        });
    });
}

fn progress_steps(event: &Value) -> Vec<Step> {
    event["steps"]
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
        .collect()
}

fn show_report(window: &AppWindow, report: &Value, ram_total: f32) {
    let cards = recommendation_cards(report, ram_total);
    if cards.is_empty() {
        let why: Vec<String> = report["no_results_reason"].as_array().into_iter().flatten().map(text).collect();
        let why = if why.is_empty() { "No model fits these needs.".to_string() } else { why.join(" ") };
        window.set_search_status(why.into());
    } else {
        window.set_search_status(format!("{} models fit this computer, best first.", cards.len()).into());
    }
    window.set_selected(0);
    window.set_recs(ModelRc::from(Rc::new(VecModel::from(cards))));
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

/// GPU and RAM as Jaull's hardware fit counts them: what the model takes in each pool against
/// what that pool has free. It is the physical use, without Jaull's reserve and margin: Jaull
/// calls a model that fits only without its reserve "tight", not "does not fit", and the bar
/// must not say otherwise. On unified memory there is one pool, so only RAM is shown.
struct Pools {
    has_gpu: bool,
    gpu_used: f32,
    gpu_cap: f32,
    ram_used: f32,
    ram_cap: f32,
    placement: String,
}

fn memory_pools(fit: &Value, total_bytes: Option<f64>, ram_total: f32) -> Pools {
    let g = |key: &str| fit[key].as_f64().map(|b| (b / GIB) as f32);
    let used = |pool: &str| g(&format!("{pool}_physical_bytes")).or_else(|| g(&format!("{pool}_required_bytes")));
    let gpu_used = used("gpu").unwrap_or(0.0);
    let unified = fit["memory_topology"].as_str() == Some("unified_memory");
    let has_gpu = gpu_used > 0.0 && !unified;
    let placement = match (fit["mode"].as_str(), fit["gpu_transformer_blocks"].as_u64(), fit["total_transformer_blocks"].as_u64()) {
        (Some("gpu_offload"), Some(on_gpu), Some(all)) => {
            format!("{on_gpu} of {all} layers on the GPU, the rest in RAM (slower).")
        }
        (Some("gpu_resident"), _, _) if has_gpu => "All of it on the GPU.".into(),
        _ if unified => "The GPU and the processor share the RAM, so it is one pool.".into(),
        (Some("cpu_ram") | Some("cpu_memory_pressure"), _, _) => "On the processor, in RAM.".into(),
        _ => String::new(),
    };
    // Reports without a hardware fit (or estimated without free memory) fall back to the total.
    let ram_used = used("ram")
        .filter(|_| fit.is_object())
        .unwrap_or_else(|| if has_gpu { 0.0 } else { total_bytes.map_or(0.0, |b| (b / GIB) as f32) });
    Pools {
        has_gpu,
        gpu_used,
        gpu_cap: g("available_vram_bytes").unwrap_or(0.0),
        ram_used,
        ram_cap: g("available_ram_bytes").unwrap_or(ram_total),
        placement,
    }
}

/// One card per recommendation in the report of `jaull recommend --json`.
fn recommendation_cards(report: &Value, ram_total: f32) -> Vec<Rec> {
    report["recommendations"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|rec| {
            let quant = rec["artifact"]["quantization"].as_str().unwrap_or("");
            let needs = gib(&rec["memory"]["memory"]["total_bytes"]);
            let context = rec["memory"]["inference_configuration"]["context_length"].as_u64();
            let detail = match (needs, context) {
                (Some(n), Some(c)) => format!("Needs about {n} with a {}-token context", thousands(c)),
                (Some(n), None) => format!("Needs about {n}"),
                _ => "Memory not estimated".to_string(),
            };
            let confidence = rec["confidence"].as_str().unwrap_or("unknown");
            let total_bytes = rec["memory"]["memory"]["total_bytes"].as_f64();
            let pools = memory_pools(&rec["memory"]["hardware_fit"], total_bytes, ram_total);
            let lines = |key: &str| -> String {
                rec[key].as_array().into_iter().flatten().map(|r| format!("· {}", text(r))).collect::<Vec<_>>().join("\n")
            };
            let runtime = rec["runtime_assessment"]["runtime"].as_str().unwrap_or("llama.cpp");
            let format = rec["artifact"]["format"].as_str().unwrap_or("gguf").to_uppercase();
            Rec {
                repo: SharedString::from(text(&rec["repo_id"])),
                quant: SharedString::from(quant),
                // The server starts with the context this card's memory was estimated for.
                context: SharedString::from(context.map(|c| c.to_string()).unwrap_or_default()),
                rank: SharedString::from(rec["rank"].as_u64().map_or("?".into(), |r| r.to_string())),
                name: SharedString::from(text(&rec["repo_id"])),
                fit: SharedString::from(fit_words(rec["compatibility"].as_str().unwrap_or(""))),
                fit_key: SharedString::from(rec["compatibility"].as_str().unwrap_or("unknown")),
                total: SharedString::from(total_bytes.map_or("memory unknown".into(), |b| format!("{:.2} GiB", b / GIB))),
                total_gib: total_bytes.map_or(0.0, |b| (b / GIB) as f32),
                stack: SharedString::from(format!("{runtime} · {format} · {quant}")),
                detail: SharedString::from(format!("{detail} · estimate confidence: {confidence}")),
                reason: SharedString::from(rec["reasons"].as_array().and_then(|r| r.first()).map(text).unwrap_or_default()),
                reasons: SharedString::from(lines("reasons")),
                warnings: SharedString::from(lines("warnings")),
                has_gpu: pools.has_gpu,
                gpu_used: pools.gpu_used,
                gpu_cap: pools.gpu_cap,
                ram_used: pools.ram_used,
                ram_cap: pools.ram_cap,
                placement: SharedString::from(pools.placement),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::memory_pools;
    use serde_json::json;

    #[test]
    fn a_partly_offloaded_model_shows_both_pools() {
        let fit = json!({
            "mode": "gpu_offload", "memory_topology": "discrete_memory",
            "gpu_transformer_blocks": 18, "total_transformer_blocks": 28,
            "gpu_physical_bytes": 7_516_192_768u64, "ram_physical_bytes": 3_221_225_472u64,
            "gpu_required_bytes": 8_053_063_680u64, "ram_required_bytes": 3_758_096_384u64,
            "available_vram_bytes": 8_589_934_592u64, "available_ram_bytes": 12_884_901_888u64,
        });
        let pools = memory_pools(&fit, None, 16.0);
        assert!(pools.has_gpu);
        assert_eq!((pools.gpu_used, pools.gpu_cap, pools.ram_used, pools.ram_cap), (7.0, 8.0, 3.0, 12.0));
        assert!(pools.placement.starts_with("18 of 28 layers"));
    }

    #[test]
    fn unified_memory_is_one_pool() {
        let fit = json!({
            "mode": "cpu_ram", "memory_topology": "unified_memory",
            "gpu_physical_bytes": null, "ram_physical_bytes": 5_368_709_120u64, "ram_required_bytes": 5_905_580_032u64,
            "available_vram_bytes": null, "available_ram_bytes": null,
        });
        let pools = memory_pools(&fit, None, 16.0);
        assert!(!pools.has_gpu);
        assert_eq!((pools.ram_used, pools.ram_cap), (5.0, 16.0));
    }
}
