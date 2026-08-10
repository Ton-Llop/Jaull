"""Ports for external command execution."""

from __future__ import annotations

from typing import Protocol

from jaull.domain.execution import ExecutionRequest, ExecutionResult


class ExecutionBackendProtocol(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


__all__ = ["ExecutionBackendProtocol"]
