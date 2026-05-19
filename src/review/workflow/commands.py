"""Command registry for the review pipeline CLI."""

from __future__ import annotations

from argparse import Namespace
from typing import Callable, Dict

from src.review.workflow.legacy_handlers import LEGACY_COMMAND_HANDLERS
from src.review.workflow.stage_handlers import PRIMARY_COMMAND_HANDLERS


CommandHandler = Callable[[Namespace], int]


COMMAND_HANDLERS: Dict[str, CommandHandler] = {
    **PRIMARY_COMMAND_HANDLERS,
    **LEGACY_COMMAND_HANDLERS,
}
