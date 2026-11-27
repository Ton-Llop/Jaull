//! Downloaded files, for Set up a local server and Free disk space (`jaull storage`).

use std::rc::Rc;

use slint::{ComponentHandle, ModelRc, VecModel};

use crate::format::{gib, text, GIB};
use crate::{jaull, AppWindow, LocalModel};

/// Reads the downloaded files (`jaull storage list --json`).
pub fn refresh(window: &AppWindow) {
    window.set_storage_busy(true);
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let result = jaull::run_json(&["storage", "list", "--json"]);
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
                    total / GIB,
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
pub fn remove(window: &AppWindow, args: Vec<String>) {
    window.set_storage_busy(true);
    window.set_storage_status("Removing…".into());
    let weak = window.as_weak();
    std::thread::spawn(move || {
        let mut all = vec!["storage".to_string(), "remove".to_string()];
        all.extend(args);
        all.push("--json".into());
        let refs: Vec<&str> = all.iter().map(String::as_str).collect();
        let result = jaull::run_json(&refs);
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            refresh(&window);
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
