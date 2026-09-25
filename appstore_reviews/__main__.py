"""Command-line interface for one-off review collection."""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import AppStoreReviews, ModelLoadError
from .analysis_pipeline import load_reviews, recalculate_saved_metrics, run_full_pipeline
from .insights import run_saved_insights

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
    analyze = commands.add_parser("analyze", help="run or resume the NLP scan on saved reviews")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)
    analyze.add_argument("--batch-size", type=int, default=8)
    analyze.add_argument("--concurrency", type=int, default=10)
    analyze.add_argument("--requests-per-minute", type=int, default=120)
    analyze.add_argument("--max-cost-usd", type=float)
    analyze.add_argument("--skip-local-nlp", action="store_true", help="run only LLM and deterministic stages")
    metrics = commands.add_parser("recalculate-nlp-metrics", help="recompute metrics from a saved scan")
    metrics.add_argument("--output-dir", required=True, type=Path)
    insights = commands.add_parser(
        "generate-insights",
        help="Generate actionable insights from a completed NLP scan",
    )
    insights.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory containing the output of analyze step (required)",
    )
    insights.add_argument(
        "--max-issues",
        type=int,
        default=8,
        help="Maximum number of issues to include in insights (default: 8)",
    )
    insights.add_argument(
        "--max-feature-requests",
        type=int,
        default=5,
        help="Maximum number of feature requests to include (default: 5)",
    )
    insights.add_argument(
        "--max-cost-usd",
        type=float,
        help="Maximum allowable cost in USD for LLM requests (optional)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    try:
        if args.mode == "generate-insights":
            result = run_saved_insights(
                args.output_dir,
                max_issues=args.max_issues,
                max_feature_requests=args.max_feature_requests,
                max_cost_usd=args.max_cost_usd,
            )

            summary = {
                "model": result["model"],
                "issue_insight_count": len(
                    result["issue_insights"]
                ),
                "feature_request_insight_count": len(
                    result["feature_request_insights"]
                ),
                "overall_summary": result[
                    "overall_summary"
                ],
            }

            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            return 0
        if args.mode == "recalculate-nlp-metrics":
            result = recalculate_saved_metrics(args.output_dir)
            print(json.dumps(result["analysis_summary"], ensure_ascii=True, indent=2))
            return 0
        if args.mode == "analyze":
            reviews = load_reviews(args.input)
            result = run_full_pipeline(reviews, args.output_dir,
                run_local_nlp=not args.skip_local_nlp, batch_size=args.batch_size,
                concurrency=args.concurrency, requests_per_minute=args.requests_per_minute,
                max_cost_usd=args.max_cost_usd)
            summary = {"analysis_summary": result["nlp_metrics"]["analysis_summary"],
                       "coverage_metrics": result["nlp_metrics"]["coverage_metrics"],
                       "api_usage": result["api_usage"]}
            print(json.dumps(summary, ensure_ascii=True, indent=2))
            return 1 if summary["coverage_metrics"]["error_count"] else 0
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
