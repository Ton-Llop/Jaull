//! Chat: talks to llama-server's OpenAI-compatible endpoint, streamed, with no browser.

use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};

use crate::format::thousands;
use crate::server::{Endpoint, Server};
use crate::{AppWindow, Msg};

/// Which answer is current. Stop moves it on, so a reply that was stopped (its thread may
/// still be waiting for the server) never writes into the next one.
#[derive(Default)]
pub struct Turn(AtomicU64);

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

/// What the server said, piece by piece. `keep_going` false ends it early (Stop): dropping
/// the response closes the connection, and llama-server stops generating.
fn read_chat_stream(reader: impl BufRead, mut keep_going: impl FnMut() -> bool, mut on_text: impl FnMut(&str)) -> Result<(), String> {
    for line in reader.lines() {
        if !keep_going() {
            return Ok(());
        }
        let line = line.map_err(|e| format!("The model response was interrupted: {e}"))?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(());
        }
        let chunk: Value = serde_json::from_str(data).map_err(|_| "The server sent an invalid streaming response.".to_string())?;
        if !chunk["error"].is_null() {
            return Err("The server reported an error during generation.".into());
        }
        if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str() {
            on_text(text);
        }
    }
    Err("The model response ended before it was complete.".into())
}

/// A rough token count: about 3 characters per token, a little under what llama.cpp's
/// tokenizers give for English, so the estimate errs on the side of leaving room.
fn tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(3) + 8
}

/// The latest turns that fit the context, leaving a quarter of it (at least 512 tokens) for
/// the answer. The newest message is always sent. Returns how many old messages were left out.
fn fit_history(history: &[(String, String)], context: u64) -> (&[(String, String)], usize) {
    if context == 0 || history.is_empty() {
        return (history, 0);
    }
    let budget = context.saturating_sub((context / 4).max(512));
    let mut used = 0;
    let mut start = history.len();
    while start > 0 {
        let cost = tokens(&history[start - 1].1);
        if start < history.len() && used + cost > budget {
            break;
        }
        used += cost;
        start -= 1;
    }
    // Start on a question, not on an answer whose question was left out.
    while start > 0 && start < history.len() && history[start].0 != "user" {
        start += 1;
    }
    (&history[start..], start)
}

fn set_last(window: &AppWindow, text: SharedString) {
    let rows = window.get_messages();
    let last = rows.row_count().saturating_sub(1);
    rows.set_row_data(last, Msg { mine: false, text });
}

/// The longest message the box keeps: about 10,000 tokens, more than the smallest context
/// Jaull offers, and short enough that the window can always draw it.
const MAX_MESSAGE_CHARS: usize = 30_000;

/// Called as the message is written or pasted: cuts a message that is too long, and says so.
pub fn cap_input(window: &AppWindow) {
    let text = window.get_chat_input();
    if text.chars().count() > MAX_MESSAGE_CHARS {
        let kept: String = text.chars().take(MAX_MESSAGE_CHARS).collect();
        window.set_chat_input(kept.into());
        window.set_chat_note(format!("A message can have up to {} characters; the rest was cut.", thousands(MAX_MESSAGE_CHARS as u64)).into());
    }
}

/// Sends the conversation to llama-server and streams the answer into the last bubble.
pub fn send(window: &AppWindow, server: &Arc<Server>, turn: &Arc<Turn>, message: &str) {
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
    let context = server.context_per_user.load(Ordering::SeqCst);
    let (sent, left_out) = fit_history(&history, context);
    window.set_chat_note(
        if left_out > 0 {
            format!("The {left_out} oldest messages are no longer sent: they do not fit the model's {}-token context.", thousands(context))
        } else {
            String::new()
        }
        .into(),
    );
    let messages: Vec<Value> = sent.iter().map(|(role, content)| serde_json::json!({"role": role, "content": content})).collect();
    let mine = turn.0.fetch_add(1, Ordering::SeqCst) + 1;
    let weak = window.as_weak();
    let server = Arc::clone(server);
    let turn = Arc::clone(turn);
    let question = message.to_string();
    std::thread::spawn(move || {
        let current = || turn.0.load(Ordering::SeqCst) == mine;
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
            Ok(Ok(response)) => read_chat_stream(BufReader::new(response.into_reader()), current, |piece| {
                answer.push_str(piece);
                let shown = answer.clone();
                let weak = weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = weak.upgrade() {
                        set_last(&window, shown.into());
                    }
                });
            })
            .err(),
            Ok(Err(ureq::Error::Status(code, _))) => Some(format!("The server answered with error {code}.")),
            Ok(Err(e)) => Some(format!("Could not reach the model: {e}")),
        };
        // A stopped reply was already settled by stop_reply.
        if !current() {
            return;
        }
        if error.is_none() {
            let mut kept = server.history.lock().unwrap();
            kept.push(("user".into(), question));
            kept.push(("assistant".into(), answer));
        }
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = weak.upgrade() else { return };
            if let Some(error) = error {
                set_last(&window, error.into());
            }
            window.set_replying(false);
        });
    });
}

/// Stop: keeps what was written so far, as part of the conversation, and frees the input.
pub fn stop_reply(window: &AppWindow, server: &Server, turn: &Turn) {
    if !window.get_replying() {
        return;
    }
    turn.0.fetch_add(1, Ordering::SeqCst);
    let rows = window.get_messages();
    let n = rows.row_count();
    let text = |i: usize| rows.row_data(i).map(|m| m.text.to_string()).unwrap_or_default();
    if n >= 2 {
        let answer = text(n - 1);
        let answer = if answer == "…" { String::new() } else { answer };
        let mut kept = server.history.lock().unwrap();
        kept.push(("user".into(), text(n - 2)));
        kept.push(("assistant".into(), answer.clone()));
        let shown = if answer.is_empty() { "(stopped)".to_string() } else { format!("{answer} (stopped)") };
        set_last(window, shown.into());
    }
    window.set_replying(false);
}

#[cfg(test)]
mod tests {
    use super::{fit_history, read_chat_stream};

    const CHUNK: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\r\n\r\n";

    #[test]
    fn complete_stream_emits_text() {
        let event_stream = format!("{CHUNK}data: [DONE]\r\n\r\n");
        let mut answer = String::new();
        read_chat_stream(event_stream.as_bytes(), || true, |piece| answer.push_str(piece)).unwrap();
        assert_eq!(answer, "hello");
    }

    #[test]
    fn incomplete_and_error_streams_fail() {
        assert!(read_chat_stream(CHUNK.as_bytes(), || true, |_| {}).is_err());
        let error_stream = b"data: {\"error\":{\"message\":\"generation failed\"}}\n\n";
        assert!(read_chat_stream(error_stream.as_slice(), || true, |_| {}).is_err());
    }

    #[test]
    fn stop_ends_the_stream_without_an_error() {
        let event_stream = format!("{CHUNK}{CHUNK}");
        let pieces = std::cell::Cell::new(0);
        let result = read_chat_stream(event_stream.as_bytes(), || pieces.get() == 0, |_| pieces.set(pieces.get() + 1));
        assert!(result.is_ok());
        assert_eq!(pieces.get(), 1);
    }

    fn turn(role: &str, chars: usize) -> (String, String) {
        (role.to_string(), "x".repeat(chars))
    }

    #[test]
    fn history_keeps_the_latest_turns_that_fit_and_starts_on_a_question() {
        // 2,000-token context: 1,488 for the conversation; each 1,200-char message is ~408.
        let history = vec![turn("user", 1200), turn("assistant", 1200), turn("user", 1200), turn("assistant", 1200), turn("user", 1200)];
        let (sent, left_out) = fit_history(&history, 2000);
        assert_eq!((sent.len(), left_out), (3, 2));
        assert_eq!(sent[0].0, "user");
        // Unknown context: everything goes.
        assert_eq!(fit_history(&history, 0).1, 0);
        // The newest message is sent even when it alone is too long.
        let long = vec![turn("user", 100), turn("assistant", 100), turn("user", 30_000)];
        assert_eq!(fit_history(&long, 2000).0.len(), 1);
    }
}
