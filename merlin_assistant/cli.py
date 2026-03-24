# merlin_assistant/cli.py

"""
MERLIN CLI entry point.

Installed as 'merlin' console_script via pyproject.toml.
Dispatches to: merlin init | merlin (run).
"""

import argparse
import sys


def main():
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="merlin",
        description="MERLIN — Deterministic Autonomous AI Assistant",
    )
    sub = parser.add_subparsers(dest="command")

    # merlin init
    init_parser = sub.add_parser(
        "init",
        help="Interactive setup wizard — configure MERLIN for first use",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration",
    )

    # Parse only the subcommand; everything else passes through to main.main()
    args, remaining = parser.parse_known_args()

    if args.command == "init":
        from merlin_assistant.setup_wizard import run_init
        run_init(force=args.force)
    else:
        # Build argparse Namespace matching what main.main() expects
        run_parser = argparse.ArgumentParser()
        run_parser.add_argument("--voice", action="store_true")
        run_parser.add_argument("--hybrid", action="store_true")
        run_parser.add_argument("--ui", action="store_true")
        run_parser.add_argument("--telegram", action="store_true")
        run_args = run_parser.parse_args(remaining)

        from main import main as run_merlin
        run_merlin(run_args)


if __name__ == "__main__":
    main()
