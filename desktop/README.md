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
  - `jaull recommend --json` with the TUI wizard's answers (`--use-case`, `--mode`, `--priority`,
    `--language` for the listed languages, `--other-language` for the rest, `--concurrency`,
    `--document-scale` for document work, `--commercial-use`): JSON lines, `progress` events while
    the search runs and a final `result` (the same report the TUI exports) or `error`. Cancelling
    ends the process tree.
  - `jaull serve start <repo> -q <quant> --users N --context T [--shared-context] [--lan] --json
    --no-opencode-config`: JSON lines while the model is resolved, downloaded and verified, the
    server plan, then `started` (URL, network URL, the path of the API key file, the certificate to
    trust when it uses HTTPS, the firewall rule Jaull suggests and the context per person; the key
    itself is never printed) or `error`. An error has `code` (the exit code) and, when llama.cpp is
    what is missing, `reason: "runtime_missing"` and the `backend` to install, so the app does not
    read the message.
  - `jaull serve status --json` (with the certificate when the server uses HTTPS and the context
    per person) and `jaull serve stop`.
  - `jaull storage list --json` and `jaull storage remove <repo> <file> | --evaluation-logs --json`:
    what the TUI's Free disk space screen does.
  - `jaull inspect`, `jaull estimate` and `jaull doctor`, read as text for Advanced tools.
- **The chat is native.** It talks to llama-server's OpenAI-compatible endpoint
  (`/v1/chat/completions`, streamed), so no browser is opened. Over plain HTTP on this PC, or over
  HTTPS when the server is shared: then it trusts only Jaull's own self-signed certificate (which
  also covers 127.0.0.1), never the system's certificate store.
- **A model the window started is stopped when the window closes**, so it does not keep gigabytes
  of memory busy. A server started elsewhere (TUI, CLI) is used but left running. A download or a
  start still running is cancelled, from its **Cancel** button or by closing the window; if
  llama-server was already launching, `jaull serve stop` runs so it does not stay behind.

## Screens

It follows the look of the demo on jaull.ai, which draws the TUI a little better: the site's
palette, Departure Mono, a `JAULL | <screen>` header with a breadcrumb, `[ Button ]` actions and a
status line. Everything is flat shapes and text with no animation, so an idle window does not redraw.

1. **Home**: the logo, the TUI's four entries with the same words (Guided analysis, Set up a local
   server, Advanced tools, Free disk space), a Chat entry while a model runs, and the TUI's shark
   breaching under the menu (its own eight frames from `shark_art.py`, via the website's
   `breach.json`; it jumps for ~2.6 s, then the water stays quiet for 5.2 s).
2. **Hardware**: what `jaull scan` detects.
3. **Your needs**: the TUI wizard's seven questions, with its options, labels and defaults: use,
   how requests are processed, priority, languages (four listed plus Other), people at once, text
   size (only for document work, numbered as the TUI does) and commercial use.
4. **Results**: the search checklist while it runs (with Cancel and the TUI's shark swimming),
   then the ranked list with memory bars per model, GPU and RAM apart as Jaull's hardware fit
   counts them (what the model takes in each pool against what that pool has free; one RAM bar on
   unified memory), and a detail pane: **Use this model** (this PC,
   for the chat, with the context of the estimate), **Serve it for other people** (people at once,
   context per person, shared context, local network, as in the demo's Serve view), fit, required
   memory, reasons and warnings.
5. **Chat**: the server's URLs, the API key (hidden until asked), the firewall rule when shared, the
   conversation, **Stop** to end an answer halfway (what was written stays in the conversation),
   **Cancel** while a model downloads or starts, and **Stop model** to free its memory. Only the
   latest turns that fit the server's context per person are sent (a rough count of 3 characters
   per token, leaving a quarter for the answer), and the window says how many were left out. The
   message box wraps: Enter sends, Shift+Enter adds a line.
6. **Set up a local server**: pick a downloaded model, people at once, context per person and who
   can reach it, then start it. A file whose quantization Jaull does not recognise is shown but
   cannot be started from here.
7. **Advanced tools**: Scan (opens Hardware), Inspect, Estimate and Doctor, which run the CLI and
   show its plain-text report.
8. **Free disk space**: the downloaded files and the evaluation logs with their sizes; **Remove**
   asks for confirmation on the same row. Saved results are never removed.

The shark is the same four frames as the website's GIF (`ui/shark/`, 148×141, 140 ms each). Slint
does not play GIFs, so a `Timer` changes frame and position; it exists only while a search is shown.

## Fonts

`ui/fonts/DepartureMono-Regular.otf` is Departure Mono 1.500 by Helena Zhang, under the SIL Open Font
License 1.1 (`ui/fonts/DepartureMono-OFL.txt`). It was converted from the website's `woff2` with
fontTools and is embedded in the executable. It has no ✓ ● ○ glyphs, so the app draws those as shapes.

## Installer

```powershell
powershell -ExecutionPolicy Bypass -File desktop\installer\build.ps1
```

It builds the app, puts a portable Python (python-build-standalone, through `uv python install
--no-bin`, so nothing lands on this PC's PATH) next to it, installs Jaull and every dependency at the
versions `uv.lock` pins, and packs both with Inno Setup (`installer\jaull.iss`). The result is
`desktop\target\installer\Jaull-Setup-<version>.exe`: 139 MiB, 0.80 GiB once installed (torch alone is
about half). It installs per user in `%LOCALAPPDATA%\Programs\Jaull` without administrator rights,
with a Start menu entry, an optional desktop shortcut and an uninstaller. The app finds that Python
next to `Jaull.exe` before anything else. Needs Rust, uv and Inno Setup 6 (`winget install
JRSoftware.InnoSetup`).

Tested on this laptop: a silent install into a test folder (87 s), the bundled Python running
`jaull scan`, the installed app starting that Python and idling at 10.3 MiB, and a clean uninstall
(no folder or Programs entry left). llama.cpp is not bundled: on a PC without it, starting a model
offers **Install llama.cpp**, which runs `jaull runtime install` for the backend Jaull names. The
installer is not code-signed, so Windows SmartScreen will warn the first time.

## Code

`src/main.rs` only connects the window to the modules: `jaull.rs` (starting the engine, ending its
processes), `format.rs`, `hardware.rs` (Hardware), `search.rs` (Your needs and Results), `server.rs`
(start, cancel, stop, llama.cpp install), `chat.rs`, `storage.rs` and `tools.rs` (Advanced tools).
The window is `ui/app.slint`, with the colours in `ui/theme.slint` and the small pieces (buttons,
options, fields, memory bars) in `ui/components.slint`. `cargo test` runs the Rust tests.

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
| This app, software renderer, demo style with the embedded font | 10 MiB | 26 MiB | 1 |
| The first version, default Slint widgets | 7 MiB | 24 MiB | 1 |
| Same hello-world in Slint with the default OpenGL renderer | 109 MiB | 118 MiB | 1 |
| Same hello-world in Tauri (WebView2) | 138 MiB | 346 MiB | 7 |

While it works, the engine is the heavy part: `jaull scan` peaks at about 120 MiB of private
memory for 3 s, and a recommendation search at about 770 MiB for 40 s to 1.5 minutes (97 s the
first time, 39 s the second, with Jaull's cache). Both end when they finish; after a search the
window was back at 9 MiB. With Qwen2.5-0.5B loaded and two chat turns, the window stayed at
9 MiB and llama-server used 618 MiB. These are single measurements on one machine, not benchmarks.

While the shark swims the window uses about 1 MiB more (10.5 MiB private) and its CPU time over 5 s
stayed below what this measurement resolves (0.00 s); after the search it is back to idle.

## Tested on this laptop

Hardware scan; a real search and its cancellation (no Python process left); **Use this model** with a
download (MiniCPM5-2B, 146 s) and without one (13 s); the chat over HTTP and over HTTPS (a server
started with `--https`); **Serve** for 2 people with a shared 32K pool (`--parallel 2`, unified KV);
**Stop model**; closing the window stops the model the window started. Also checked on 2026-11-26:
the wizard's command line for document work with another language; **Cancel** during a download and
once llama-server was already starting (nothing left running, `serve status` not running); closing
the window during a download (it exits in 0.3 s, no process left); **Stop** halfway through an
answer (llama-server stops generating, and the next question sees the partial answer); a 20,000
character message followed by another one, with the first left out to fit an 8,192-token context. The search checklist was
checked against a fake Jaull with slow steps, because a cached real search goes through them too fast
to see. **Sharing on the local network was not tested**: it opens a port, and Windows may ask about
the firewall.

## Not done

- The installer is not signed, and it was tested on one PC only.
- No Markdown in the chat, no saved conversations. The token count for the history is a rough
  estimate, not the model's tokenizer.
- The **Install llama.cpp** button was not tried on a PC without llama.cpp.
- The interface is in English, like the CLI and the TUI.
