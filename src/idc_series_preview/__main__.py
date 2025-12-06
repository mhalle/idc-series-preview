"""Entry point delegating to the Click CLI."""

from __future__ import annotations

from .cli_click import cli


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
