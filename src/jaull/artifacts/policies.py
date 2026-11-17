"""Limits for checking a local file against what Hugging Face has published."""

# How many of the repository's latest commits are searched for a local file's digest. Each
# one is a request, and only a file whose digest differs from today's pays for them. Past
# this the file is treated as unknown rather than as an old version.
ARTIFACT_HISTORY_COMMIT_LIMIT = 50
