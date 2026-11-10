"""Thread-local cancellation for synchronous execution chains."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from jaull.execution.errors import ExecutionError

_check: ContextVar[Callable[[], bool] | None] = ContextVar("execution_cancel", default=None)


class ExecutionCancelledError(ExecutionError):
    """The caller cancelled its own execution."""


def check_cancelled() -> None:
    check = _check.get()
    if check is not None and check():
        raise ExecutionCancelledError("Execution cancelled.")


@contextmanager
def cancellation_scope(check: Callable[[], bool]) -> Iterator[None]:
    token = _check.set(check)
    try:
        check_cancelled()
        yield
    finally:
        _check.reset(token)
