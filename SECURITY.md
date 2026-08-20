# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private vulnerability reporting instead: go to the repository's **Security**
tab and choose *Report a vulnerability*. That creates a private advisory visible only to
the maintainers.

Useful things to include: what the issue is, how to reproduce it, the affected version or
commit, and what an attacker could do with it.

## Scope

Jaull runs on a user's own machine. It talks to the Hugging Face Hub over HTTPS, writes
artifacts and records under the per-user data directory, and — only when explicitly asked —
downloads model files and launches local runtime processes such as `llama-cli`.

Reports that are in scope include, for example: path traversal or unsafe writes outside the
artifact store, artifact verification that can be bypassed, credential or token leakage
(`HF_TOKEN` in logs, reports or exported files), and command construction that lets
untrusted metadata influence a local process invocation.

Out of scope: the security of the models themselves, of `llama.cpp`, PyTorch or any other
third-party runtime, and of the Hugging Face Hub. Report those to their own projects.

## Expectations

This is a small project maintained by one person alongside other work, so there is no
guaranteed response time and no bug bounty. Reports are taken seriously and will be
acknowledged as soon as reasonably possible.

## Supported versions

Jaull has not reached a stable release. Fixes land on `master`; there are no maintained
release branches yet.
