"""Approval package - human-in-the-loop state machine."""
from .workflow import (
    ApprovalWorkflow,
    STATES,
    TRANSITIONS,
    get_workflow,
    set_workflow,
)

__all__ = [
    "ApprovalWorkflow",
    "STATES",
    "TRANSITIONS",
    "get_workflow",
    "set_workflow",
]
