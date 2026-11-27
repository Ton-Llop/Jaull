//! Hardware screen: what `jaull scan --json` detects.

use std::rc::Rc;

use serde_json::Value;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};

use crate::format::{gib, text, GIB};
use crate::{jaull, AppWindow, Row};

/// Runs the scan in the background and fills the Hardware screen and the header.
pub fn start_scan(window: &AppWindow) {
    window.set_busy(true);
    window.set_status("Detecting hardware… (a few seconds)".into());
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let result = jaull::run_json(&["scan", "--json"]).map(|v| {
            let hw = &v["hardware"];
            (hardware_rows(hw), machine_summary(hw), hw["memory"]["total_bytes"].as_f64().unwrap_or(0.0))
        });
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            match result {
                Ok((rows, machine, ram_bytes)) => {
                    window.set_machine(machine.into());
                    if ram_bytes > 0.0 {
                        window.set_ram_total_gib((ram_bytes / GIB) as f32);
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
    if let Some(acc) = hw["accelerators"].as_array().and_then(|a| a.first()) {
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
    }
    parts.join(" · ")
}
