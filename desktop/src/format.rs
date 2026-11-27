//! Small helpers that turn Jaull's JSON values into the words the window shows.

use serde_json::Value;

pub const GIB: f64 = 1_073_741_824.0;

/// Bytes as "4.2 GiB"; None when the value is missing.
pub fn gib(value: &Value) -> Option<String> {
    value.as_f64().map(|bytes| format!("{:.1} GiB", bytes / GIB))
}

/// A string field, or "?" when it is missing.
pub fn text(value: &Value) -> String {
    value.as_str().unwrap_or("?").to_string()
}

/// 32768 as "32,768".
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
