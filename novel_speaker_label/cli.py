from __future__ import annotations

import argparse
from pathlib import Path

from .annotation import AnnotationConfig, annotate_volume
from .discovery import DiscoveryConfig, discover_volume
from .preprocess import PreprocessConfig, find_volume_file, preprocess_volume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="novel_speaker_label",
        description="Stage 0/1/2 pipeline for light-novel speaker labeling.",
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
    discover_parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Socket timeout in seconds. Use 0 to disable it for long local runs.",
    )
    discover_parser.add_argument("--temperature", type=float, default=0.0)
    discover_parser.add_argument("--num-predict", type=int, default=4096)
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
    discover_parser.add_argument(
        "--max-paragraphs-per-request",
        type=int,
        default=30,
        help="Maximum paragraphs sent to Ollama in one discovery request.",
    )
    discover_parser.add_argument(
        "--max-dialogues-per-request",
        type=int,
        default=40,
        help="Maximum dialogue units sent to Ollama in one discovery request.",
    )
    discover_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one discovery request fails.",
    )
    discover_parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only read existing discovery/cache JSON files; never call Ollama.",
    )

    annotate_parser = subparsers.add_parser(
        "annotate", help="Stage 2: label each dialogue speaker from memory files."
    )
    _add_common_volume_args(annotate_parser, require_input=False)
    annotate_parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="Ollama model to use. Repeat for multi-model voting.",
    )
    annotate_parser.add_argument(
        "--pipeline",
        choices=("vote", "structured"),
        default="vote",
        help="Stage 2 pipeline. structured runs evidence -> judge -> contradiction.",
    )
    annotate_parser.add_argument(
        "--evidence-model",
        dest="evidence_models",
        action="append",
        default=None,
        help="Model for structured evidence extraction. Repeat to ensemble.",
    )
    annotate_parser.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        default=None,
        help="Model for structured adjudication. Repeat to ensemble.",
    )
    annotate_parser.add_argument(
        "--contradiction-model",
        dest="contradiction_models",
        action="append",
        default=None,
        help="Model for structured contradiction checks. Repeat to ensemble.",
    )
    annotate_parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    annotate_parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Socket timeout in seconds. Use 0 to disable it for long local runs.",
    )
    annotate_parser.add_argument("--temperature", type=float, default=0.0)
    annotate_parser.add_argument("--num-predict", type=int, default=2048)
    annotate_parser.add_argument("--dry-run", action="store_true")
    annotate_parser.add_argument("--overwrite-cache", action="store_true")
    annotate_parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only read existing annotation/cache JSON files; never call Ollama.",
    )
    annotate_parser.add_argument("--stop-on-error", action="store_true")
    annotate_parser.add_argument(
        "--model-weight",
        action="append",
        default=[],
        metavar="MODEL=WEIGHT",
        help="Optional aggregation weight for a model. Repeat as needed.",
    )
    annotate_parser.add_argument("--max-characters", type=int, default=12)
    annotate_parser.add_argument("--max-mysteries", type=int, default=8)
    annotate_parser.add_argument("--max-scene-summaries", type=int, default=6)
    annotate_parser.add_argument(
        "--annotation-window-size",
        type=int,
        default=8,
        help="Number of same-scene dialogue lines sent in one annotation request.",
    )
    annotate_parser.add_argument(
        "--max-dialogue-paragraph-gap",
        type=int,
        default=4,
        help="Split annotation windows when neighboring dialogues are farther apart.",
    )
    annotate_parser.add_argument(
        "--context-paragraph-radius",
        type=int,
        default=3,
        help="Paragraphs before/after each dialogue window sent as local context.",
    )
    annotate_parser.add_argument(
        "--max-window-paragraphs",
        type=int,
        default=48,
        help="Maximum context paragraphs included in one annotation request.",
    )
    annotate_parser.add_argument(
        "--scene-summary-radius",
        type=int,
        default=0,
        help="Nearby stage-1 scene-memory chunks to include around the dialogue window.",
    )
    annotate_parser.add_argument(
        "--start-dialogue-index",
        type=int,
        default=0,
        help="Zero-based dialogue_index to start from.",
    )
    annotate_parser.add_argument(
        "--dialogue-limit",
        type=int,
        default=None,
        help="Maximum number of dialogues to process.",
    )
    annotate_parser.add_argument("--min-confidence", type=float, default=0.75)
    annotate_parser.add_argument("--min-agreement", type=float, default=0.65)
    annotate_parser.add_argument("--min-margin", type=float, default=0.20)
    annotate_parser.add_argument("--min-support-models", type=int, default=2)

    rebuild_parser = subparsers.add_parser(
        "rebuild-memory",
        help="Rebuild scene_discoveries and memory files from existing cache only.",
    )
    _add_common_volume_args(rebuild_parser, require_input=False)
    rebuild_parser.add_argument("--model", default="qwen3:32b")
    rebuild_parser.add_argument(
        "--max-paragraphs-per-request",
        type=int,
        default=30,
        help="Must match the request split used when the cache was created.",
    )
    rebuild_parser.add_argument(
        "--max-dialogues-per-request",
        type=int,
        default=40,
        help="Must match the request split used when the cache was created.",
    )
    rebuild_parser.add_argument("--stop-on-error", action="store_true")

    run_parser = subparsers.add_parser(
        "run-volume", help="Run stage 0 followed by stage 1 for one volume."
    )
    _add_common_volume_args(run_parser)
    run_parser.add_argument("--model", default="qwen3:32b")
    run_parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Socket timeout in seconds. Use 0 to disable it for long local runs.",
    )
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--num-predict", type=int, default=4096)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--overwrite-cache", action="store_true")
    run_parser.add_argument("--max-paragraphs-per-request", type=int, default=30)
    run_parser.add_argument("--max-dialogues-per-request", type=int, default=40)
    run_parser.add_argument("--stop-on-error", action="store_true")

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
                max_paragraphs_per_request=args.max_paragraphs_per_request,
                max_dialogues_per_request=args.max_dialogues_per_request,
                continue_on_error=not args.stop_on_error,
                cache_only=args.cache_only,
            )
        )
        _print_summary("discover", summary)
        return 0

    if args.command == "annotate":
        try:
            model_weights = _parse_model_weights(args.model_weight)
        except ValueError as exc:
            parser.error(str(exc))
        summary = annotate_volume(
            AnnotationConfig(
                output_dir=_resolve_output(args),
                models=tuple(args.models or ["qwen3:32b"]),
                pipeline=args.pipeline,
                evidence_models=tuple(args.evidence_models or ()),
                judge_models=tuple(args.judge_models or ()),
                contradiction_models=tuple(args.contradiction_models or ()),
                ollama_host=args.ollama_host,
                timeout=args.timeout,
                temperature=args.temperature,
                num_predict=args.num_predict,
                dry_run=args.dry_run,
                overwrite_cache=args.overwrite_cache,
                cache_only=args.cache_only,
                continue_on_error=not args.stop_on_error,
                model_weights=model_weights,
                max_characters=args.max_characters,
                max_mysteries=args.max_mysteries,
                max_scene_summaries=args.max_scene_summaries,
                annotation_window_size=args.annotation_window_size,
                max_dialogue_paragraph_gap=args.max_dialogue_paragraph_gap,
                context_paragraph_radius=args.context_paragraph_radius,
                max_window_paragraphs=args.max_window_paragraphs,
                scene_summary_radius=args.scene_summary_radius,
                start_dialogue_index=args.start_dialogue_index,
                dialogue_limit=args.dialogue_limit,
                min_confidence=args.min_confidence,
                min_agreement=args.min_agreement,
                min_margin=args.min_margin,
                min_support_models=args.min_support_models,
            )
        )
        _print_summary("annotate", summary)
        return 0

    if args.command == "rebuild-memory":
        summary = discover_volume(
            DiscoveryConfig(
                output_dir=_resolve_output(args),
                model=args.model,
                max_paragraphs_per_request=args.max_paragraphs_per_request,
                max_dialogues_per_request=args.max_dialogues_per_request,
                continue_on_error=not args.stop_on_error,
                cache_only=True,
                write_prompts=False,
            )
        )
        _print_summary("rebuild-memory", summary)
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
                max_paragraphs_per_request=args.max_paragraphs_per_request,
                max_dialogues_per_request=args.max_dialogues_per_request,
                continue_on_error=not args.stop_on_error,
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root output directory used when --output is omitted.",
    )
    parser.set_defaults(require_input=require_input)


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input:
        return args.input
    return find_volume_file(args.novels_dir, args.volume)


def _resolve_output(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return args.output_root / f"volume_{args.volume:02d}"


def _print_summary(stage: str, summary: dict) -> None:
    print(f"[{stage}]")
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"{key}: {value}")


def _parse_model_weights(rows: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"Invalid --model-weight {row!r}; expected MODEL=WEIGHT")
        model, raw_weight = row.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid --model-weight {row!r}; model is empty")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --model-weight {row!r}; weight must be a number"
            ) from exc
        weights[model] = weight
    return weights
