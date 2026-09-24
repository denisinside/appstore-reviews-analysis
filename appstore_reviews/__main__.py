"""Command-line interface for one-off review collection."""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import AppStoreReviews, ModelLoadError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m appstore_reviews")
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("discovery", "country", "top"):
        command = commands.add_parser(name)
        command.add_argument("--app-id", required=True)
        command.add_argument("--force-refresh", action="store_true")
        command.add_argument("--output", type=Path)
        if name == "country":
            command.add_argument("--country", required=True)
            command.add_argument("--max-pages", type=int, default=10)
        elif name == "top":
            command.add_argument("--top", type=int, default=10)
            command.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    try:
        with AppStoreReviews() as scraper:
            if args.mode == "discovery":
                result = scraper.discovery(args.app_id, force_refresh=args.force_refresh)
            elif args.mode == "country":
                result = scraper.get_reviews(
                    args.app_id, args.country, max_pages=args.max_pages, force_refresh=args.force_refresh
                )
            else:
                result = scraper.get_top_reviews(
                    args.app_id, top_n=args.top, max_pages=args.max_pages,
                    force_refresh=args.force_refresh,
                )
        # ASCII escapes keep stdout valid even on Windows legacy code pages.
        rendered = json.dumps(result, ensure_ascii=not bool(args.output), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 1 if result.get("errors") or result.get("status") in ("partial", "failed") else 0
    except (ValueError, ModelLoadError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
