"""Small domain contracts for running external commands.

These objects deliberately know nothing about subprocesses, llama.cpp, Docker
or Hugging Face. They describe an execution request and the observed result.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = Field(ge=0.0)


class InferenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    duration_seconds: float = Field(ge=0.0)
    exit_code: int
    runtime: str
    model_path: Path


__all__ = ["ExecutionRequest", "ExecutionResult", "InferenceResult"]
