# Jaull desktop (experiment)

A light Windows window over the existing Jaull engine, for people who will not open a terminal.
It is an experiment on the `desktop-app-test` branch: it does not replace the TUI or the CLI, and it
does not change how Jaull estimates or recommends. It only calls the CLI and shows what it returns.

## Design

- **The window is native and small.** It is written in Rust with [Slint](https://slint.dev) and uses
  Slint's software renderer, which does not load the graphics driver into the process.
- **The engine runs only while it works.** Each action starts `python -m jaull <command> --json`
  and the process ends when the answer is back. When the window is idle nothing else stays in memory.
- **The contract is the CLI's JSON**:
  - `jaull scan --json`: the hardware profile.
  - `jaull recommend --json`: JSON lines, `progress` events while the search runs and a final
    `result` (the same report the TUI exports) or `error`. Cancelling ends the process tree.
  - `jaull serve start <repo> -q <quant> --json --no-opencode-config`: JSON lines while the model
    is resolved, downloaded and verified, the server plan, then `started` (URL and the path of the
    API key file; the key itself is never printed) or `error`.
  - `jaull serve status --json` and `jaull serve stop`.
- **The chat is native.** It talks to llama-server's OpenAI-compatible endpoint
  (`/v1/chat/completions`, streamed) over plain HTTP on 127.0.0.1, so no browser is opened.
- **A model the window started is stopped when the window closes**, so it does not keep gigabytes
  of memory busy. A server started elsewhere (TUI, CLI) is used but left running.

## Screens

It follows the look of the demo on jaull.ai, which draws the TUI a little better: the site's
palette, Departure Mono, a `JAULL | <screen>` header with a breadcrumb, `[ Button ]` actions and a
status line. Everything is flat shapes and text with no animation, so an idle window does not redraw.

1. **Home**: the logo and two entries, the guided analysis and the chat.
2. **Hardware**: what `jaull scan` detects.
3. **Your needs**: the wizard's questions as numbered radio lists, plus languages.
4. **Results**: the search checklist while it runs (with Cancel), then the ranked list with a
   memory bar per model and a detail pane (fit, required memory, reasons, warnings) with
   **Use this model**.
5. **Chat**: talk to the running model; **Stop model** frees its memory.

## Fonts

`ui/fonts/DepartureMono-Regular.otf` is Departure Mono 1.500 by Helena Zhang, under the SIL Open Font
License 1.1 (`ui/fonts/DepartureMono-OFL.txt`). It was converted from the website's `woff2` with
fontTools and is embedded in the executable. It has no ✓ ● ○ glyphs, so the app draws those as shapes.

## Build and run

Needs Rust (MSVC toolchain) and the Visual Studio C++ Build Tools, and this repository's `.venv`
with Jaull installed.

```powershell
cd desktop
cargo run --release
```

The executable is `desktop\target\release\jaull-desktop.exe`. It looks for Jaull in this order:
the `JAULL_PYTHON` environment variable (a Python with Jaull installed), the repository's
`.venv\Scripts\python.exe`, and `jaull` on `PATH`. `SLINT_BACKEND` can override the renderer.

## Measurements (one laptop, Ryzen 5 5500U, 2026-11-26)

Memory of the whole process tree, idle, 15 s after opening:

| | Private memory | Working set | Processes |
|---|---:|---:|---:|
| This app, software renderer | 7 MiB | 24 MiB | 1 |
| Same hello-world in Slint with the default OpenGL renderer | 109 MiB | 118 MiB | 1 |
| Same hello-world in Tauri (WebView2) | 138 MiB | 346 MiB | 7 |

While it works, the engine is the heavy part: `jaull scan` peaks at about 120 MiB of private
memory for 3 s, and a recommendation search at about 770 MiB for 40 s to 1.5 minutes (97 s the
first time, 39 s the second, with Jaull's cache). Both end when they finish; after a search the
window was back at 9 MiB. With Qwen2.5-0.5B loaded and two chat turns, the window stayed at
9 MiB and llama-server used 618 MiB. These are single measurements on one machine, not benchmarks.

## Not done

- No installer: it needs the repository and its virtualenv.
- **Use this model** has the same code path as `jaull serve start --json`, which was tested from
  the CLI (download, plan, start); the button itself has not been clicked on a real download yet.
- No way yet to stop a reply halfway, no Markdown in the chat, no saved conversations.
- The interface is in English, like the CLI and the TUI.
