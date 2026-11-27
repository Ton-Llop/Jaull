// Jaull desktop, experiment: a light window over the existing Python engine.
// The engine runs only while it is needed (one `python -m jaull ... --json` process per
// action), so nothing but this window stays in memory when the app is idle.
//
// This file only connects the window (ui/app.slint) to the modules that do the work:
//   jaull     starting the engine and ending its processes
//   format    JSON values as words (GiB, thousands)
//   hardware  the Hardware screen (jaull scan)
//   search    Your needs and Results (jaull recommend)
//   server    starting, cancelling and stopping the model (jaull serve), installing llama.cpp
//   chat      the conversation with llama-server
//   storage   downloaded files (jaull storage)
//   tools     Advanced tools (jaull inspect, estimate, doctor)
#![windows_subsystem = "windows"]

mod chat;
mod format;
mod hardware;
mod jaull;
mod search;
mod server;
mod storage;
mod tools;

use std::sync::Arc;

use slint::ComponentHandle;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    // The software renderer keeps the window at a few MiB; the default OpenGL one loaded
    // the graphics driver and used about 110 MiB in the hello-world measurement.
    if std::env::var_os("SLINT_BACKEND").is_none() {
        std::env::set_var("SLINT_BACKEND", "winit-software");
    }
    let window = AppWindow::new()?;
    let search = Arc::new(search::Search::default());
    let server = Arc::new(server::Server::default());
    let turn = Arc::new(chat::Turn::default());

    // Each callback gets a weak handle to the window (and the state it needs) and does nothing
    // if the window is already gone.
    macro_rules! on {
        ($setter:ident, |$window:ident $(, $arg:ident)*| $body:expr) => {{
            let weak = window.as_weak();
            window.$setter(move |$($arg),*| {
                if let Some($window) = weak.upgrade() {
                    $body
                }
            });
        }};
    }

    on!(on_rescan, |w| hardware::start_scan(&w));

    let s = Arc::clone(&search);
    on!(on_find, |w| search::start_search(&w, &s));
    let s = Arc::clone(&search);
    on!(on_cancel, |w| if s.cancel() {
        w.set_search_status("Cancelling…".into());
    });

    let s = Arc::clone(&server);
    on!(on_use_model, |w, repo, quant, context| {
        let request = server::ServeRequest { users: 1, context: context.to_string(), shared: false, lan: false };
        server::use_model(&w, &s, &repo, &quant, request);
    });
    let s = Arc::clone(&server);
    on!(on_serve_model, |w, repo, quant, users, context, shared, lan| {
        let request = server::ServeRequest { users, context: context.to_string(), shared, lan };
        server::use_model(&w, &s, &repo, &quant, request);
    });
    let s = Arc::clone(&server);
    on!(on_cancel_start, |w| server::cancel_start(&w, &s));
    let s = Arc::clone(&server);
    on!(on_stop_model, |w| server::stop_model(&w, &s));
    on!(on_install_runtime, |w, backend| server::install_runtime(&w, &backend));

    let (s, t) = (Arc::clone(&server), Arc::clone(&turn));
    on!(on_send, |w, message| chat::send(&w, &s, &t, &message));
    let (s, t) = (Arc::clone(&server), Arc::clone(&turn));
    on!(on_stop_reply, |w| chat::stop_reply(&w, &s, &t));
    on!(on_chat_edited, |w| chat::cap_input(&w));

    on!(on_refresh_storage, |w| storage::refresh(&w));
    on!(on_remove_model, |w, repo, filename| storage::remove(&w, vec![repo.to_string(), filename.to_string()]));
    on!(on_remove_logs, |w| storage::remove(&w, vec!["--evaluation-logs".to_string()]));
    on!(on_run_tool, |w, tool, repo, quant, context| tools::run_tool(&w, tool, &repo, &quant, &context));

    hardware::start_scan(&window);
    server::check_server(&window, &server);
    let result = window.run();
    // Closing the window must not leave a search running in the background, a download or
    // start half done, nor a model this window started holding gigabytes of memory.
    search.cancel();
    server::shut_down(&server);
    result
}
