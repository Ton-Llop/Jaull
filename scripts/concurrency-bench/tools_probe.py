"""Does an OpenAI-compatible llama-server return real tool calls?

A coding agent such as opencode edits files and runs commands through tool calls. If the
server cannot produce them, the agent just answers with text. This sends one request with a
tool and checks that the answer carries a tool call. Standard library only.

    python3 scripts/concurrency-bench/tools_probe.py --url http://127.0.0.1:8090 --api-key KEY

Exit code: 0 a tool call came back, 1 it did not, 2 the server could not be asked.
The key is only sent as the Authorization header; it is never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

REQUEST = {
    "max_tokens": 200,
    "temperature": 0,
    "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }}],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="local")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    body = json.dumps({**REQUEST, "model": args.model}).encode()
    request = urllib.request.Request(
        f"{args.url}/v1/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            choice = json.load(response)["choices"][0]
    except (OSError, urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"Could not ask the server: {type(exc).__name__}: {exc}")
        return 2

    message = choice.get("message", {})
    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0]["function"]
        print(f"OK: tool call {function['name']}({function['arguments']}), "
              f"finish_reason={choice.get('finish_reason')}")
        return 0
    print(f"NO TOOL CALL: finish_reason={choice.get('finish_reason')}, answered with text: "
          f"{(message.get('content') or '')[:120]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
