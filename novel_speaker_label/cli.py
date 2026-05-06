from __future__ import annotations

import argparse
from pathlib import Path

from .discovery import DiscoveryConfig, discover_volume
from .preprocess import PreprocessConfig, find_volume_file, preprocess_volume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="novel_speaker_label",
        description="Stage 0/1 pipeline for light-novel speaker labeling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Stage 0: split a volume into structured units."
    )
    _add_common_volume_args(preprocess_parser)
    preprocess_parser.add_argument(
        "--context-paragraphs",
        type=int,
        default=3,
        help="Number of neighboring paragraphs stored around each dialogue.",
    )
    preprocess_parser.add_argument(
        "--max-scene-paragraphs",
        type=int,
        default=80,
        help="Maximum paragraphs per heuristic scene chunk.",
    )
    preprocess_parser.add_argument(
        "--no-source-copy",
        action="store_true",
        help="Do not copy the source txt into the output directory.",
    )

    discover_parser = subparsers.add_parser(
        "discover", help="Stage 1: discover characters and build memory files."
    )
    _add_common_volume_args(discover_parser, require_input=False)
    discover_parser.add_argument("--model", default="qwen3:32b")
    discover_parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    discover_parser.add_argument("--timeout", type=int, default=120)
    discover_parser.add_argument("--temperature", type=float, default=0.0)
    discover_parser.add_argument("--num-predict", type=int, default=8192)
    discover_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts only; do not call Ollama.",
    )
    discover_parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Ignore cached scene discovery responses.",
    )
    discover_parser.add_argument(
        "--max-known-characters",
        type=int,
        default=30,
        help="Number of recently discovered character cards sent into the next prompt.",
    )

    run_parser = subparsers.add_parser(
        "run-volume", help="Run stage 0 followed by stage 1 for one volume."
    )
    _add_common_volume_args(run_parser)
    run_parser.add_argument("--model", default="qwen3:32b")
    run_parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    run_parser.add_argument("--timeout", type=int, default=120)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--num-predict", type=int, default=8192)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--overwrite-cache", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "preprocess":
        input_path = _resolve_input(args)
        output_dir = _resolve_output(args)
        metadata = preprocess_volume(
            PreprocessConfig(
                input_path=input_path,
                output_dir=output_dir,
                volume=args.volume,
                context_paragraphs=args.context_paragraphs,
                max_scene_paragraphs=args.max_scene_paragraphs,
                copy_source=not args.no_source_copy,
            )
        )
        _print_summary("preprocess", metadata)
        return 0

    if args.command == "discover":
        summary = discover_volume(
            DiscoveryConfig(
                output_dir=_resolve_output(args),
                model=args.model,
                ollama_host=args.ollama_host,
                timeout=args.timeout,
                temperature=args.temperature,
                num_predict=args.num_predict,
                dry_run=args.dry_run,
                overwrite_cache=args.overwrite_cache,
                max_known_characters=args.max_known_characters,
            )
        )
        _print_summary("discover", summary)
        return 0

    if args.command == "run-volume":
        input_path = _resolve_input(args)
        output_dir = _resolve_output(args)
        preprocess_metadata = preprocess_volume(
            PreprocessConfig(
                input_path=input_path,
                output_dir=output_dir,
                volume=args.volume,
            )
        )
        _print_summary("preprocess", preprocess_metadata)
        discovery_summary = discover_volume(
            DiscoveryConfig(
                output_dir=output_dir,
                model=args.model,
                ollama_host=args.ollama_host,
                timeout=args.timeout,
                temperature=args.temperature,
                num_predict=args.num_predict,
                dry_run=args.dry_run,
                overwrite_cache=args.overwrite_cache,
            )
        )
        _print_summary("discover", discovery_summary)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_common_volume_args(
    parser: argparse.ArgumentParser, require_input: bool = True
) -> None:
    parser.add_argument("--volume", type=int, default=1, help="Volume number.")
    parser.add_argument(
        "--input",
        type=Path,
        required=False,
        help="Path to the source txt. If omitted, it is inferred from --novels-dir.",
    )
    parser.add_argument(
        "--novels-dir",
        type=Path,
        default=Path("novels"),
        help="Directory containing source novel txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/volume_XX.",
    )
    parser.set_defaults(require_input=require_input)


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input:
        return args.input
    return find_volume_file(args.novels_dir, args.volume)


def _resolve_output(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return Path("outputs") / f"volume_{args.volume:02d}"


def _print_summary(stage: str, summary: dict) -> None:
    print(f"[{stage}]")
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"{key}: {value}")
