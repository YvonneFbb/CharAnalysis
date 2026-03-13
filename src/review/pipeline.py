"""
Review pipeline entrypoint.

Thin wrapper:
- parser definition lives in `src.review.workflow.parser`
- command handlers live in `src.review.workflow.commands`
"""

from __future__ import annotations

import sys

from src.review.workflow.commands import COMMAND_HANDLERS
from src.review.workflow.parser import create_parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())

