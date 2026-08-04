"""Serialise domain objects into JSON, Markdown and other export formats.

Reporting is one step above the domain and orchestration layers: it reads
already-computed estimates, recommendations and workflow states and turns
them into strings that a report file or an external consumer can read.

Never imports ``presentation/`` (Rich rendering lives there and stays there).
"""
