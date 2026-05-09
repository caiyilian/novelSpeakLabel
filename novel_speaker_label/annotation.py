from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jsonl import read_json, read_jsonl, write_json, write_jsonl
from .ollama_client import OllamaClient, OllamaConfig

SPEAKER_STATUSES = {"known", "mystery", "npc", "ambiguous", "review"}
REVIEW_LABEL = "待复核"
LOW_SIGNAL_NAMES = {"咱", "汝", "你", "我", "他", "她", "它", "我们", "他们", "她们"}
LOW_SIGNAL_TITLES = {
    "中间人",
    "交易者",
    "同行",
    "商人",
    "小毛头",
    "小的",
    "市井无赖",
    "年轻人",
    "开店者",
    "情报贩子",
    "提议者",
    "新手",
    "旅行商人",
    "旅行商人亲戚",
    "行商人",
    "老板",
    "贵族",
}
SPEECH_VERBS = (
    "说",
    "说道",
    "问",
    "询问",
    "回答",
    "答道",
    "喊",
    "大喊",
    "叫",
    "嚷",
    "喃喃",
    "发出声音",
)
DIRECT_ATTRIBUTION_VERBS = (*SPEECH_VERBS, "念着", "打招呼")
SELF_INTRO_MARKERS = ("我是", "我叫", "在下是", "本人是", "名字是")
RULE_MODEL = "rule:context"
RULE_MODEL_PREFIX = "rule:"
RULE_CONFIDENCE = 0.98
ALTERNATION_RULE_CONFIDENCE = 0.92
ALTERNATION_MAX_PARAGRAPH_GAP = 3
ADDRESS_TITLES = ("先生", "小姐", "大人", "阁下", "夫人")
SECOND_PERSON_MARKERS = ("你", "您")
LISTENER_REACTION_MARKERS = ("被询问", "被问", "被这么一问", "被追问", "被提问")
INDIRECT_SPEECH_SUFFIXES = ("的话", "的事", "的内容", "的说法")
VILLAGE_CONTEXT_MARKERS = ("小村落", "深山", "山里的村民", "村落", "村民")
VILLAGE_NPC_DISPLAY = "小村落的村民"


@dataclass(frozen=True)
class AnnotationConfig:
    output_dir: Path
    models: tuple[str, ...] = ("qwen3:32b",)
    ollama_host: str = "http://127.0.0.1:11434"
    timeout: int = 1800
    temperature: float = 0.0
    num_predict: int = 2048
    dry_run: bool = False
    overwrite_cache: bool = False
    cache_only: bool = False
    continue_on_error: bool = True
    write_prompts: bool = True
    max_characters: int = 12
    max_mysteries: int = 8
    max_scene_summaries: int = 6
    annotation_window_size: int = 8
    max_dialogue_paragraph_gap: int = 4
    context_paragraph_radius: int = 3
    max_window_paragraphs: int = 48
    scene_summary_radius: int = 0
    start_dialogue_index: int = 0
    dialogue_limit: int | None = None
    min_confidence: float = 0.75
    min_agreement: float = 0.65
    min_margin: float = 0.20
    min_support_models: int = 2
    model_weights: dict[str, float] = field(default_factory=dict)


def annotate_volume(config: AnnotationConfig) -> dict:
    volume_meta = read_json(config.output_dir / "volume.json")
    preprocess_dir = config.output_dir / "preprocess"
    memory_dir = config.output_dir / "memory"
    paragraphs = list(read_jsonl(preprocess_dir / "paragraphs.jsonl"))
    dialogues = _select_dialogues(
        list(read_jsonl(preprocess_dir / "dialogues.jsonl")),
        start_index=config.start_dialogue_index,
        limit=config.dialogue_limit,
    )
    paragraphs_by_id = {row["paragraph_id"]: row for row in paragraphs}
    scenes_by_id = {
        row["scene_id"]: row
        for row in _read_optional_jsonl(preprocess_dir / "scenes.jsonl")
        if row.get("scene_id")
    }

    characters = _read_optional_jsonl(memory_dir / "semantic" / "characters.jsonl")
    mysteries = _read_optional_jsonl(memory_dir / "mystery_entities.jsonl")
    scene_memories = _read_optional_jsonl(memory_dir / "episodic" / "scenes.jsonl")
    scene_memories_by_parent = _group_scene_memories(scene_memories)
    dialogue_windows = _dialogue_windows(
        dialogues,
        config.annotation_window_size,
        max_paragraph_gap=config.max_dialogue_paragraph_gap,
    )

    annotation_dir = config.output_dir / "annotation"
    cache_dir = annotation_dir / "cache"
    prompt_dir = annotation_dir / "prompts"
    raw_dir = annotation_dir / "raw"
    failure_dir = annotation_dir / "failures"
    for path in (cache_dir, prompt_dir, raw_dir, failure_dir):
        path.mkdir(parents=True, exist_ok=True)

    clients = {
        model: OllamaClient(
            OllamaConfig(
                host=config.ollama_host,
                model=model,
                timeout=config.timeout,
                temperature=config.temperature,
                num_predict=config.num_predict,
            )
        )
        for model in config.models
    }

    votes: list[dict] = []
    annotations: list[dict] = []
    failed_requests: list[dict] = []
    prompt_count = 0
    rule_vote_count = 0

    for request_number, dialogue_window in enumerate(dialogue_windows, start=1):
        first_dialogue = dialogue_window[0]
        last_dialogue = dialogue_window[-1]
        print(
            "[annotate] "
            f"{request_number}/{len(dialogue_windows)} "
            f"{first_dialogue['dialogue_id']}..{last_dialogue['dialogue_id']} "
            f"scene={first_dialogue['scene_id']}",
            flush=True,
        )
        payload = _build_annotation_payload(
            dialogues=dialogue_window,
            paragraphs=paragraphs,
            paragraphs_by_id=paragraphs_by_id,
            scenes_by_id=scenes_by_id,
            scene_memories_by_parent=scene_memories_by_parent,
            characters=characters,
            mysteries=mysteries,
            volume_meta=volume_meta,
            config=config,
        )
        dialogue_votes_by_id: dict[str, list[dict]] = {
            dialogue["dialogue_id"]: [] for dialogue in dialogue_window
        }

        for model in config.models:
            request_id = f"{_dialogue_window_id(dialogue_window)}--{_safe_model_name(model)}"
            prompt_path = prompt_dir / f"{request_id}.txt"
            prompt = ""
            if config.write_prompts or config.dry_run:
                prompt = _build_annotation_prompt(payload)
                prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
            prompt_count += 1

            cached_path = cache_dir / f"{request_id}.json"
            if config.dry_run:
                continue

            try:
                if cached_path.exists() and not config.overwrite_cache:
                    parsed_response = read_json(cached_path)
                    response_text = json.dumps(parsed_response, ensure_ascii=False)
                elif config.cache_only:
                    raise FileNotFoundError(f"Cached annotation vote not found: {cached_path}")
                else:
                    if not prompt:
                        prompt = _build_annotation_prompt(payload)
                    response_text = clients[model].generate(prompt)
                    (raw_dir / f"{request_id}.txt").write_text(
                        response_text, encoding="utf-8", newline="\n"
                    )
                    parsed_response = _parse_json_response(response_text)
                    write_json(cached_path, parsed_response)
            except Exception as exc:
                failure = {
                    "request_id": request_id,
                    "dialogue_ids": [row["dialogue_id"] for row in dialogue_window],
                    "model": model,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "prompt_path": str(prompt_path),
                }
                write_json(failure_dir / f"{request_id}.json", failure)
                failed_requests.append(failure)
                print(
                    "[annotate] failed "
                    f"{request_id}: {failure['error_type']}: {failure['error']}",
                    flush=True,
                )
                if not config.continue_on_error:
                    raise RuntimeError(
                        f"Annotation request failed: {request_id}. "
                        f"Prompt is saved at {prompt_path}"
                    ) from exc
                continue

            parsed_votes = _extract_parsed_votes(parsed_response, dialogue_window)
            for dialogue in dialogue_window:
                parsed_vote = parsed_votes.get(dialogue["dialogue_id"])
                if parsed_vote is None:
                    parsed_vote = _missing_dialogue_vote(dialogue, request_id)
                vote = _normalize_vote(
                    parsed_vote=parsed_vote,
                    dialogue=dialogue,
                    model=model,
                    weight=config.model_weights.get(model, 1.0),
                )
                votes.append(vote)
                dialogue_votes_by_id[dialogue["dialogue_id"]].append(vote)

        if not config.dry_run:
            rule_votes = _rule_votes_for_window(dialogue_window, payload)
            rule_vote_count += len(rule_votes)
            for vote in rule_votes:
                votes.append(vote)
                dialogue_votes_by_id[vote["dialogue_id"]].append(vote)

            for dialogue in dialogue_window:
                annotations.append(
                    _aggregate_votes(
                        dialogue,
                        dialogue_votes_by_id[dialogue["dialogue_id"]],
                        config,
                    )
                )

    if config.dry_run:
        write_json(
            annotation_dir / "dry_run.json",
            {
                "volume_id": volume_meta["volume_id"],
                "models": list(config.models),
                "dialogue_count": len(dialogues),
                "window_count": len(dialogue_windows),
                "annotation_window_size": max(1, config.annotation_window_size),
                "max_dialogue_paragraph_gap": max(0, config.max_dialogue_paragraph_gap),
                "prompt_count": prompt_count,
                "rule_vote_count": 0,
                "message": "Prompts were generated; no Ollama requests were made.",
            },
        )
    else:
        write_jsonl(annotation_dir / "votes.jsonl", votes)
        write_jsonl(annotation_dir / "annotations.jsonl", annotations)
        write_jsonl(
            annotation_dir / "review_queue.jsonl",
            [row for row in annotations if row["needs_review"]],
        )
        write_jsonl(annotation_dir / "failed_requests.jsonl", failed_requests)
        _write_labeled_text(
            paragraphs=paragraphs,
            annotations=annotations,
            output_path=annotation_dir / "final_labeled.txt",
        )

    review_count = sum(1 for row in annotations if row.get("needs_review"))
    summary = {
        "volume_id": volume_meta["volume_id"],
        "models": list(config.models),
        "dry_run": config.dry_run,
        "cache_only": config.cache_only,
        "dialogue_count": len(dialogues),
        "window_count": len(dialogue_windows),
        "annotation_window_size": max(1, config.annotation_window_size),
        "max_dialogue_paragraph_gap": max(0, config.max_dialogue_paragraph_gap),
        "request_count": prompt_count,
        "prompt_count": prompt_count,
        "rule_vote_count": rule_vote_count,
        "vote_count": len(votes),
        "annotation_count": len(annotations),
        "review_count": review_count,
        "failed_request_count": len(failed_requests),
    }
    write_json(annotation_dir / "run_summary.json", summary)
    primary_output = annotation_dir / ("dry_run.json" if config.dry_run else "annotations.jsonl")
    print(
        "[annotate] wrote "
        f"{_output_path_for_log(primary_output)} "
        f"annotations={summary['annotation_count']} "
        f"review={summary['review_count']} "
        f"failures={summary['failed_request_count']}",
        flush=True,
    )
    return summary


def _select_dialogues(
    dialogues: list[dict], start_index: int, limit: int | None
) -> list[dict]:
    selected = [
        row for row in dialogues if int(row.get("dialogue_index", 0)) >= start_index
    ]
    if limit is not None:
        return selected[: max(0, limit)]
    return selected


def _dialogue_windows(
    dialogues: list[dict], window_size: int, max_paragraph_gap: int = 4
) -> list[list[dict]]:
    max_size = max(1, int(window_size))
    max_gap = max(0, int(max_paragraph_gap))
    windows: list[list[dict]] = []
    current: list[dict] = []
    current_scene_id = ""
    for dialogue in dialogues:
        scene_id = _clean_text(dialogue.get("scene_id"))
        if current and (
            scene_id != current_scene_id
            or len(current) >= max_size
            or _dialogue_gap_exceeds(current[-1], dialogue, max_gap)
        ):
            windows.append(current)
            current = []
        current.append(dialogue)
        current_scene_id = scene_id
    if current:
        windows.append(current)
    return windows


def _dialogue_gap_exceeds(previous: dict, current: dict, max_gap: int) -> bool:
    if max_gap <= 0:
        return False
    try:
        previous_index = int(previous.get("paragraph_index"))
        current_index = int(current.get("paragraph_index"))
    except (TypeError, ValueError):
        return False
    return current_index - previous_index > max_gap


def _dialogue_window_id(dialogues: list[dict]) -> str:
    if len(dialogues) == 1:
        return dialogues[0]["dialogue_id"]
    return f"{dialogues[0]['dialogue_id']}_to_{dialogues[-1]['dialogue_id']}"


def _read_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _group_scene_memories(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        parent_id = row.get("parent_scene_id") or row.get("scene_id")
        if not parent_id:
            continue
        grouped.setdefault(parent_id, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row.get("chunk_index", 0)))
    return grouped


def _build_annotation_payload(
    dialogues: list[dict],
    paragraphs: list[dict],
    paragraphs_by_id: dict[str, dict],
    scenes_by_id: dict[str, dict],
    scene_memories_by_parent: dict[str, list[dict]],
    characters: list[dict],
    mysteries: list[dict],
    volume_meta: dict,
    config: AnnotationConfig,
) -> dict:
    scene_id = dialogues[0]["scene_id"]
    context_rows = _window_context_rows(
        dialogues=dialogues,
        paragraphs=paragraphs,
        paragraphs_by_id=paragraphs_by_id,
        config=config,
    )
    scene_memories = _select_scene_memories_for_window(
        dialogues=dialogues,
        scene_meta=scenes_by_id.get(scene_id, {}),
        scene_memories=scene_memories_by_parent.get(scene_id, []),
        config=config,
    )
    active_names = _active_names(scene_memories)
    context_text = _window_context_text(dialogues, context_rows, scene_memories)
    context_ids = {row.get("paragraph_id") for row in context_rows if row.get("paragraph_id")}
    context_ids.update(dialogue["dialogue_id"] for dialogue in dialogues)
    scene_memory_ids = {
        _clean_text(row.get("scene_id")) for row in scene_memories if row.get("scene_id")
    }

    return {
        "volume": {
            "volume_id": volume_meta["volume_id"],
            "volume": volume_meta.get("volume"),
        },
        "target_dialogue_ids": [dialogue["dialogue_id"] for dialogue in dialogues],
        "dialogues": [
            _dialogue_card(dialogue, window_position=index + 1)
            for index, dialogue in enumerate(dialogues)
        ],
        "context": {
            "paragraphs": _paragraph_context_cards(context_rows, dialogues),
        },
        "scene_memory": [_scene_card(row) for row in scene_memories],
        "candidate_characters": [
            _character_card(row)
            for row in _select_character_candidates(
                characters=characters,
                active_names=active_names,
                context_text=context_text,
                scene_id=scene_id,
                scene_memory_ids=scene_memory_ids,
                limit=config.max_characters,
            )
        ],
        "candidate_mysteries": [
            _mystery_card(row)
            for row in _select_mystery_candidates(
                mysteries=mysteries,
                context_ids=context_ids,
                context_text=context_text,
                scene_id=scene_id,
                scene_memory_ids=scene_memory_ids,
                limit=config.max_mysteries,
            )
        ],
    }


def _dialogue_card(dialogue: dict, window_position: int) -> dict:
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "dialogue_index": dialogue["dialogue_index"],
        "window_position": window_position,
        "chapter_id": dialogue["chapter_id"],
        "chapter_title": dialogue["chapter_title"],
        "scene_id": dialogue["scene_id"],
        "paragraph_id": dialogue["paragraph_id"],
        "paragraph_index": dialogue["paragraph_index"],
        "local_dialogue_index": dialogue["local_dialogue_index"],
        "text": dialogue["text"],
        "quote_text": dialogue["quote_text"],
        "paragraph_text": dialogue["paragraph_text"],
        "dialogue_kind": _dialogue_kind(dialogue),
        "outside_quote_text": _outside_quote_text(dialogue),
        "char_start": dialogue["char_start"],
        "char_end": dialogue["char_end"],
    }


def _dialogue_kind(dialogue: dict) -> str:
    kind = _clean_text(dialogue.get("dialogue_kind"))
    if kind:
        return kind
    return "inline" if _outside_quote_text(dialogue) else "standalone"


def _outside_quote_text(dialogue: dict) -> str:
    paragraph_text = (
        "" if dialogue.get("paragraph_text") is None else str(dialogue.get("paragraph_text"))
    )
    if not paragraph_text:
        return ""
    try:
        char_start = int(dialogue.get("char_start", 0))
        char_end = int(dialogue.get("char_end", char_start))
    except (TypeError, ValueError):
        return ""
    char_start = max(0, min(char_start, len(paragraph_text)))
    char_end = max(char_start, min(char_end, len(paragraph_text)))
    return _clean_text(paragraph_text[:char_start] + paragraph_text[char_end:])


def _window_context_rows(
    dialogues: list[dict],
    paragraphs: list[dict],
    paragraphs_by_id: dict[str, dict],
    config: AnnotationConfig,
) -> list[dict]:
    if not dialogues:
        return []
    paragraph_indexes = [
        int(dialogue.get("paragraph_index", 0))
        for dialogue in dialogues
        if dialogue.get("paragraph_index") is not None
    ]
    if not paragraph_indexes:
        return _dedupe_context_rows(
            _dialogue_context_rows(dialogue, paragraphs_by_id.get(dialogue["paragraph_id"], {}))
            for dialogue in dialogues
        )

    start = max(0, min(paragraph_indexes) - max(0, config.context_paragraph_radius))
    end = min(
        len(paragraphs) - 1,
        max(paragraph_indexes) + max(0, config.context_paragraph_radius),
    )
    if end >= start and end - start + 1 <= max(1, config.max_window_paragraphs):
        return [paragraphs[index] for index in range(start, end + 1)]

    return _dedupe_context_rows(
        _dialogue_context_rows(dialogue, paragraphs_by_id.get(dialogue["paragraph_id"], {}))
        for dialogue in dialogues
    )[: max(1, config.max_window_paragraphs)]


def _dedupe_context_rows(row_groups: Any) -> list[dict]:
    rows: dict[str, dict] = {}
    for group in row_groups:
        for row in group:
            paragraph_id = _clean_text(row.get("paragraph_id"))
            if not paragraph_id or paragraph_id in rows:
                continue
            rows[paragraph_id] = row
    return sorted(rows.values(), key=lambda row: int(row.get("paragraph_index", 0)))


def _paragraph_context_cards(rows: list[dict], dialogues: list[dict]) -> list[dict]:
    dialogue_ids_by_paragraph: dict[str, list[str]] = {}
    for dialogue in dialogues:
        dialogue_ids_by_paragraph.setdefault(dialogue["paragraph_id"], []).append(
            dialogue["dialogue_id"]
        )
    return [
        {
            "paragraph_id": row.get("paragraph_id"),
            "paragraph_index": row.get("paragraph_index"),
            "text": _truncate(row.get("text", ""), 360),
            "target_dialogue_ids": dialogue_ids_by_paragraph.get(
                row.get("paragraph_id"), []
            ),
        }
        for row in rows
    ]


def _select_scene_memories_for_window(
    dialogues: list[dict],
    scene_meta: dict,
    scene_memories: list[dict],
    config: AnnotationConfig,
) -> list[dict]:
    if not scene_memories:
        return []
    chunk_count = max(
        [int(row.get("chunk_count", 0) or 0) for row in scene_memories] or [0]
    )
    if not scene_meta or chunk_count <= 1:
        return scene_memories[: max(1, config.max_scene_summaries)]

    target_chunks = {
        _estimate_scene_chunk(dialogue, scene_meta, chunk_count)
        for dialogue in dialogues
    }
    radius = max(0, config.scene_summary_radius)
    ranked: list[tuple[int, int, dict]] = []
    for index, memory in enumerate(scene_memories):
        chunk_index = int(memory.get("chunk_index", 0) or 0)
        distance = min(abs(chunk_index - chunk) for chunk in target_chunks)
        if distance <= radius:
            ranked.append((distance, index, memory))
    if not ranked:
        ranked = [
            (0, index, memory)
            for index, memory in enumerate(scene_memories[: max(1, config.max_scene_summaries)])
        ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = [row for _, _, row in ranked[: max(1, config.max_scene_summaries)]]
    return sorted(selected, key=lambda row: int(row.get("chunk_index", 0) or 0))


def _estimate_scene_chunk(dialogue: dict, scene_meta: dict, chunk_count: int) -> int:
    start = int(scene_meta.get("start_paragraph_index", dialogue.get("paragraph_index", 0)))
    end = int(scene_meta.get("end_paragraph_index", start))
    paragraph_index = int(dialogue.get("paragraph_index", start))
    total = max(1, end - start + 1)
    offset = min(max(paragraph_index - start, 0), total - 1)
    estimated = int(offset * max(1, chunk_count) / total) + 1
    return min(max(estimated, 1), max(1, chunk_count))


def _dialogue_context_rows(dialogue: dict, current_paragraph: dict) -> list[dict]:
    rows: list[dict] = []
    rows.extend(_context_from_dialogue(dialogue, "prev_context"))
    rows.append(
        {
            "paragraph_id": current_paragraph.get("paragraph_id", dialogue["paragraph_id"]),
            "text": current_paragraph.get("text", dialogue.get("paragraph_text", "")),
        }
    )
    rows.extend(_context_from_dialogue(dialogue, "next_context"))
    return rows


def _context_from_dialogue(dialogue: dict, key: str) -> list[dict]:
    rows = dialogue.get(key)
    if not isinstance(rows, list):
        return []
    return [
        {
            "paragraph_id": row.get("paragraph_id"),
            "text": row.get("text", ""),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _active_names(scene_memories: list[dict]) -> set[str]:
    names: set[str] = set()
    for row in scene_memories:
        for value in _as_list(row.get("active_characters")):
            cleaned = _clean_text(value)
            if cleaned:
                names.add(cleaned)
    return names


def _context_text(dialogue: dict, context_rows: list[dict], scene_memories: list[dict]) -> str:
    parts = [dialogue.get("text", ""), dialogue.get("paragraph_text", "")]
    parts.extend(row.get("text", "") for row in context_rows)
    for memory in scene_memories:
        parts.append(memory.get("scene_summary", ""))
        parts.extend(_as_list(memory.get("active_characters")))
        parts.extend(_as_list(memory.get("relationships")))
        parts.append(memory.get("notes", ""))
    return "\n".join(str(part) for part in parts if part)


def _window_context_text(
    dialogues: list[dict], context_rows: list[dict], scene_memories: list[dict]
) -> str:
    parts: list[Any] = []
    for dialogue in dialogues:
        parts.append(dialogue.get("text", ""))
        parts.append(dialogue.get("paragraph_text", ""))
    parts.extend(row.get("text", "") for row in context_rows)
    return "\n".join(str(part) for part in parts if part)


def _select_character_candidates(
    characters: list[dict],
    active_names: set[str],
    context_text: str,
    scene_id: str,
    scene_memory_ids: set[str],
    limit: int,
) -> list[dict]:
    scored = [
        (
            _score_character(
                row,
                active_names=active_names,
                context_text=context_text,
                scene_id=scene_id,
                scene_memory_ids=scene_memory_ids,
            ),
            index,
            row,
        )
        for index, row in enumerate(characters)
    ]
    scored = [item for item in scored if item[0] > 0.5]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[: max(0, limit)]]


def _score_character(
    character: dict,
    active_names: set[str],
    context_text: str,
    scene_id: str,
    scene_memory_ids: set[str],
) -> float:
    score = _safe_float(character.get("confidence"), 0.5) * 0.1
    display_name = _clean_text(character.get("display_name"))
    if display_name in active_names:
        score += 12.0
    if _name_appears(display_name, context_text):
        score += 8.0
    for alias in _as_list(character.get("aliases")):
        if _name_appears(_clean_text(alias), context_text):
            score += 6.0
    for title in _as_list(character.get("titles")):
        cleaned_title = _clean_text(title)
        if cleaned_title in LOW_SIGNAL_TITLES:
            continue
        if _name_appears(cleaned_title, context_text):
            score += 2.0
    if _clean_text(character.get("first_seen_scene_id")) in scene_memory_ids:
        score += 1.0
    if _clean_text(character.get("latest_seen_scene_id")) in scene_memory_ids:
        score += 1.5
    return score


def _name_appears(name: str, text: str) -> bool:
    if not name or name in LOW_SIGNAL_NAMES:
        return False
    if len(name) <= 1:
        return False
    return name in text


def _select_mystery_candidates(
    mysteries: list[dict],
    context_ids: set[str],
    context_text: str,
    scene_id: str,
    scene_memory_ids: set[str],
    limit: int,
) -> list[dict]:
    scored = [
        (
            _score_mystery(
                row,
                context_ids=context_ids,
                context_text=context_text,
                scene_id=scene_id,
                scene_memory_ids=scene_memory_ids,
            ),
            index,
            row,
        )
        for index, row in enumerate(mysteries)
    ]
    scored = [item for item in scored if item[0] >= 0]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[: max(1, limit)]]


def _score_mystery(
    mystery: dict,
    context_ids: set[str],
    context_text: str,
    scene_id: str,
    scene_memory_ids: set[str],
) -> float:
    score = _safe_float(mystery.get("confidence"), 0.5) * 0.1
    source_scene_id = _clean_text(mystery.get("source_scene_id"))
    source_parent_id = _parent_scene_id(source_scene_id)
    direct_evidence = any(
        _evidence_id(evidence) in context_ids for evidence in _as_list(mystery.get("evidence"))
    )
    temporary_name = _clean_text(mystery.get("temporary_name"))
    if temporary_name in LOW_SIGNAL_TITLES:
        return -1.0
    direct_name = _name_appears(temporary_name, context_text)

    if not (direct_evidence or direct_name):
        return -1.0
    if source_parent_id != scene_id and not direct_evidence:
        return -1.0

    if source_scene_id in scene_memory_ids:
        score += 2.0
    if direct_evidence:
        score += 8.0
    if direct_name:
        score += 4.0
    return score


def _evidence_id(value: Any) -> str:
    if isinstance(value, dict):
        return _clean_text(
            value.get("text") or value.get("paragraph_id") or value.get("dialogue_id")
        )
    return _clean_text(value)


def _parent_scene_id(scene_id: str) -> str:
    return re.sub(r"-r\d+$", "", _clean_text(scene_id))


def _scene_card(row: dict) -> dict:
    return {
        "scene_id": row.get("scene_id"),
        "parent_scene_id": row.get("parent_scene_id"),
        "chunk_index": row.get("chunk_index"),
        "chunk_count": row.get("chunk_count"),
        "scene_summary": _truncate(row.get("scene_summary", ""), 260),
        "active_characters": _unique_strings(_as_list(row.get("active_characters")), 12),
        "relationships": _unique_strings(_as_list(row.get("relationships")), 6),
        "notes": _truncate(row.get("notes", ""), 160),
    }


def _character_card(row: dict) -> dict:
    return {
        "entity_id": row.get("entity_id"),
        "display_name": row.get("display_name"),
        "aliases": _unique_strings(_as_list(row.get("aliases")), 8),
        "titles": _unique_strings(_as_list(row.get("titles")), 8),
        "description": _truncate(row.get("description", ""), 260),
        "speech_style": _truncate(row.get("speech_style", ""), 160),
        "relationship_hints": _unique_strings(
            _as_list(row.get("relationship_hints")), 8
        ),
        "first_seen_scene_id": row.get("first_seen_scene_id"),
        "latest_seen_scene_id": row.get("latest_seen_scene_id"),
        "confidence": _safe_float(row.get("confidence"), 0.5),
    }


def _mystery_card(row: dict) -> dict:
    return {
        "mystery_id": _mystery_id(row),
        "temporary_name": row.get("temporary_name"),
        "description": _truncate(row.get("description", ""), 240),
        "evidence": _unique_strings(_as_list(row.get("evidence")), 8),
        "source_scene_id": row.get("source_scene_id"),
        "confidence": _safe_float(row.get("confidence"), 0.5),
    }


def _mystery_id(row: dict) -> str:
    source = _clean_identifier(row.get("source_scene_id") or "unknown_scene")
    name = _clean_identifier(row.get("temporary_name") or "unknown")
    return f"mystery_{source}_{name}"[:160]


def _build_annotation_prompt(payload: dict) -> str:
    return (
        "你是轻小说说话人标注项目的阶段 2：正式标注助手。\n"
        "你的任务是判断输入 target_dialogues 中每一句台词的说话人，不要改写原文。\n"
        "必须把这些台词当作同一个连续片段来判断，优先保持对话轮次和叙述线索的一致性。\n\n"
        "规则：\n"
        "1. 只输出严格 JSON，不要 Markdown，不要解释性前后缀。\n"
        "2. 如果说话人是候选角色，speaker_entity_id 必须使用 candidate_characters 里的 entity_id，speaker_status 填 known。\n"
        "3. 如果说话人像某个未知人物，speaker_entity_id 使用 candidate_mysteries 里的 mystery_id，speaker_status 填 mystery。\n"
        "4. 如果只是无法追踪的路人/群体，speaker_status 填 npc，speaker_entity_id 可为空或使用 npc:简短称呼。\n"
        "5. 证据不足时不要硬猜，speaker_status 填 ambiguous 或 review，并把 needs_review 设为 true。\n"
        "6. candidate_speakers 至少列出 1 个候选；如果无法判断，列出最可能的候选和不确定原因。\n"
        "7. confidence 使用 0 到 1 之间的小数，表示你对最终判断的把握。\n"
        "8. 不要只因为台词里提到某个名字、职业、地点，就把说话人标成那个实体；被称呼的人通常不是说话人。\n"
        "9. 连续问答、交易寒暄、命令/回答等场景要按上下句轮次判断；同一人连续说话必须有叙述或语气证据支持。\n"
        "10. 如果当前上下文显示某角色还没有登场，不要因为角色库或后文记忆里有这个角色就提前标给他/她。\n\n"
        "11. target_dialogues 中 dialogue_kind=inline 的项目是正文内嵌引号，不要当作独立对白；标为 review，needs_review=true。\n"
        "12. candidate_mysteries 只是可选候选；如果候选名称和当前段落地点/人物不一致，必须改用 npc 或 review。\n"
        "13. 如果只有轮次推断、没有叙述锚点或自我介绍，请降低 confidence，并在 evidence 里说明轮次依据。\n\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "annotations": [\n'
        "    {\n"
        '      "dialogue_id": "必须来自 target_dialogue_ids",\n'
        '      "speaker_entity_id": "string",\n'
        '      "speaker_display": "string",\n'
        '      "speaker_status": "known|mystery|npc|ambiguous|review",\n'
        '      "confidence": 0.0,\n'
        '      "candidate_speakers": [\n'
        '        {"entity_id": "string", "display": "string", "status": "known|mystery|npc|ambiguous|review", "score": 0.0}\n'
        "      ],\n"
        '      "evidence": ["短证据，不要大段复制原文"],\n'
        '      "should_create_new_entity": false,\n'
        '      "new_entity_hint": "",\n'
        '      "needs_review": false\n'
        "    }\n"
        "  ],\n"
        '  "window_notes": "可选，简短说明整体轮次判断"\n'
        "}\n"
        "必须为 target_dialogue_ids 中每个 dialogue_id 输出且只输出一条 annotation。\n\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _parse_json_response(response_text: str) -> Any:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_parsed_votes(parsed_response: Any, dialogues: list[dict]) -> dict[str, dict]:
    if isinstance(parsed_response, list):
        rows = parsed_response
    elif isinstance(parsed_response, dict):
        rows = _first_list_value(
            parsed_response,
            ("annotations", "dialogues", "results", "items", "votes"),
        )
        if rows is None and parsed_response.get("dialogue_id"):
            rows = [parsed_response]
    else:
        rows = None

    if rows is None:
        rows = []

    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        dialogue_id = _clean_text(row.get("dialogue_id"))
        if dialogue_id:
            by_id[dialogue_id] = row

    if not by_id and len(dialogues) == 1 and isinstance(parsed_response, dict):
        by_id[dialogues[0]["dialogue_id"]] = parsed_response
    return by_id


def _first_list_value(row: dict, keys: tuple[str, ...]) -> list | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            return value
    return None


def _missing_dialogue_vote(dialogue: dict, request_id: str) -> dict:
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "speaker_entity_id": "review:missing_from_response",
        "speaker_display": REVIEW_LABEL,
        "speaker_status": "review",
        "confidence": 0.0,
        "candidate_speakers": [],
        "evidence": [f"模型响应 {request_id} 未包含该 dialogue_id"],
        "should_create_new_entity": False,
        "new_entity_hint": "",
        "needs_review": True,
    }


def _normalize_vote(
    parsed_vote: dict, dialogue: dict, model: str, weight: float
) -> dict:
    status = _clean_status(parsed_vote.get("speaker_status"))
    speaker_entity_id = _clean_text(parsed_vote.get("speaker_entity_id"))
    speaker_display = _clean_text(parsed_vote.get("speaker_display"))
    confidence = _clamp(_safe_float(parsed_vote.get("confidence"), 0.0), 0.0, 1.0)
    needs_review = bool(parsed_vote.get("needs_review")) or status in {
        "ambiguous",
        "review",
    }
    if not speaker_entity_id and status in {"ambiguous", "review"}:
        speaker_entity_id = f"{status}:unknown"
    if not speaker_display and status in {"ambiguous", "review"}:
        speaker_display = REVIEW_LABEL

    return {
        "dialogue_id": dialogue["dialogue_id"],
        "paragraph_id": dialogue["paragraph_id"],
        "char_start": dialogue["char_start"],
        "char_end": dialogue["char_end"],
        "model": model,
        "weight": weight,
        "speaker_entity_id": speaker_entity_id,
        "speaker_display": speaker_display,
        "speaker_status": status,
        "confidence": confidence,
        "candidate_speakers": _normalize_candidate_speakers(
            parsed_vote.get("candidate_speakers")
        ),
        "evidence": _unique_strings(_as_list(parsed_vote.get("evidence")), 8),
        "should_create_new_entity": bool(parsed_vote.get("should_create_new_entity")),
        "new_entity_hint": _clean_text(parsed_vote.get("new_entity_hint")),
        "needs_review": needs_review,
        "raw_vote": parsed_vote,
    }


def _normalize_candidate_speakers(value: Any) -> list[dict]:
    rows = value if isinstance(value, list) else []
    candidates: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidates.append(
            {
                "entity_id": _clean_text(row.get("entity_id")),
                "display": _clean_text(row.get("display") or row.get("speaker_display")),
                "status": _clean_status(row.get("status")),
                "score": _clamp(_safe_float(row.get("score"), 0.0), 0.0, 1.0),
            }
        )
    return candidates


def _rule_votes_for_window(dialogues: list[dict], payload: dict) -> list[dict]:
    anchors: dict[str, dict] = {}
    for dialogue in dialogues:
        speaker = _explicit_rule_speaker(dialogue, payload)
        if speaker:
            anchors[dialogue["dialogue_id"]] = speaker

    inferred = _infer_alternating_rule_speakers(dialogues, payload, anchors)
    speakers_by_id = {**inferred, **anchors}
    return [
        _rule_vote_for_speaker(dialogue, speakers_by_id[dialogue["dialogue_id"]])
        for dialogue in dialogues
        if dialogue["dialogue_id"] in speakers_by_id
    ]


def _explicit_rule_speaker(dialogue: dict, payload: dict) -> dict | None:
    if _dialogue_kind(dialogue) != "standalone":
        return None

    speaker = _self_intro_rule_speaker(dialogue, payload)
    if speaker:
        return speaker

    for text in _direct_attribution_contexts(dialogue):
        option = _speaker_option_from_direct_attribution(text, payload)
        if option:
            return _speaker_from_option(
                option,
                f"规则：上下文叙述把台词归给{option['display']}",
                RULE_CONFIDENCE,
            )

    speaker = _speaker_from_listener_reaction(dialogue, payload)
    if speaker:
        return speaker

    return _speaker_from_addressed_name(dialogue, payload)


def _self_intro_rule_speaker(dialogue: dict, payload: dict) -> dict | None:
    text = _clean_text(dialogue.get("text"))
    for option in _known_speaker_options(payload):
        for name in option["names"]:
            if _self_intro_mentions(text, name):
                return _speaker_from_option(
                    option,
                    f"规则：自我介绍提到{name}",
                    RULE_CONFIDENCE,
                )
    return None


def _speaker_from_listener_reaction(dialogue: dict, payload: dict) -> dict | None:
    text = _clean_text(dialogue.get("text"))
    if "？" not in text and "?" not in text:
        return None
    next_text = "\n".join(
        _clean_text(row.get("text"))
        for row in _as_list(dialogue.get("next_context"))[:2]
        if isinstance(row, dict)
    )
    if not next_text or not any(marker in next_text for marker in LISTENER_REACTION_MARKERS):
        return None

    option = _listener_option_from_reaction_text(next_text, payload)
    if option:
        counterpart = _counterpart_speaker(
            mentioned_option=option,
            payload=payload,
            dialogue=dialogue,
            context_text=_dialogue_context_text(dialogue),
            evidence=f"规则：下一句叙述显示{option['display']}是被询问的一方",
        )
        if counterpart:
            return counterpart
    return None


def _listener_option_from_reaction_text(text: str, payload: dict) -> dict | None:
    marker_positions = [
        index
        for marker in LISTENER_REACTION_MARKERS
        for index in [text.find(marker)]
        if index >= 0
    ]
    if not marker_positions:
        return None

    matches: list[tuple[int, int, dict]] = []
    for option in _known_speaker_options(payload):
        for name in option["names"]:
            for match in re.finditer(re.escape(name), text):
                nearest_marker = min(
                    marker_positions, key=lambda index: abs(index - match.start())
                )
                after_marker_penalty = 20 if match.start() > nearest_marker else 0
                distance = abs(nearest_marker - match.start()) + after_marker_penalty
                matches.append((distance, -len(name), option))
    if not matches:
        return None
    matches.sort(key=lambda row: (row[0], row[1]))
    return matches[0][2]


def _speaker_from_addressed_name(dialogue: dict, payload: dict) -> dict | None:
    text = _clean_text(dialogue.get("text"))
    if not text:
        return None
    for option in _known_speaker_options(payload):
        for name in option["names"]:
            if _name_is_addressed(text, name):
                counterpart = _counterpart_speaker(
                    mentioned_option=option,
                    payload=payload,
                    dialogue=dialogue,
                    context_text=_dialogue_context_text(dialogue),
                    evidence=f"规则：台词是在称呼{option['display']}，被称呼者不是说话人",
                )
                if counterpart:
                    return counterpart
    return None


def _name_is_addressed(text: str, name: str) -> bool:
    if not name or name in LOW_SIGNAL_NAMES:
        return False
    titled = "|".join(re.escape(title) for title in ADDRESS_TITLES)
    second_person = "|".join(re.escape(marker) for marker in SECOND_PERSON_MARKERS)
    return bool(
        re.search(
            f"{re.escape(name)}(?:{titled})?[，、。！？!\\s]*({second_person})",
            text,
        )
        or re.search(f"{re.escape(name)}(?:{titled})", text)
    )


def _counterpart_speaker(
    mentioned_option: dict,
    payload: dict,
    dialogue: dict,
    context_text: str,
    evidence: str,
) -> dict | None:
    mentioned_key = _speaker_rule_key(mentioned_option)
    mentioned_others = [
        option
        for option in _known_speaker_options(payload)
        if _speaker_rule_key(option) != mentioned_key
        and _option_mentioned_in_text(option, context_text)
    ]
    if len(mentioned_others) == 1:
        return _speaker_from_option(mentioned_others[0], evidence, RULE_CONFIDENCE)

    all_others = [
        option
        for option in _known_speaker_options(payload)
        if _speaker_rule_key(option) != mentioned_key
    ]
    if len(all_others) == 1:
        return _speaker_from_option(all_others[0], evidence, RULE_CONFIDENCE)

    npc = _npc_speaker_from_context([context_text])
    if npc:
        return _speaker_with_evidence(npc, evidence, RULE_CONFIDENCE)
    return None


def _direct_attribution_contexts(dialogue: dict) -> list[str]:
    texts = [_outside_quote_text(dialogue)]
    texts.extend(_adjacent_narration_rows(_as_list(dialogue.get("next_context"))))
    return [text for text in (_clean_text(value) for value in texts) if text]


def _adjacent_narration_rows(rows: list[Any], limit: int = 2) -> list[str]:
    texts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _clean_text(row.get("text", ""))
        if not text:
            continue
        if _looks_like_standalone_quote_text(text):
            break
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _looks_like_standalone_quote_text(text: str) -> bool:
    stripped = _clean_text(text)
    return stripped.startswith("「") and stripped.endswith("」")


def _speaker_option_from_direct_attribution(text: str, payload: dict) -> dict | None:
    matches: list[tuple[int, int, int, dict]] = []
    for option in _known_speaker_options(payload):
        for name in option["names"]:
            if not name or name not in text:
                continue
            for verb in DIRECT_ATTRIBUTION_VERBS:
                name_then_verb = f"{re.escape(name)}.{{0,30}}{re.escape(verb)}"
                for match in re.finditer(name_then_verb, text):
                    if _speech_verb_match_is_indirect(text, match.end(), verb):
                        continue
                    matches.append((match.start(), match.end(), -len(name), option))

    if not matches:
        return None
    matches.sort(key=lambda row: (row[0], row[1], row[2]))
    return matches[0][3]


def _infer_alternating_rule_speakers(
    dialogues: list[dict], payload: dict, anchors: dict[str, dict]
) -> dict[str, dict]:
    inferred: dict[str, dict] = {}
    for segment in _standalone_rule_segments(dialogues):
        anchor_items = [
            (index, anchors[dialogue["dialogue_id"]])
            for index, dialogue in enumerate(segment)
            if dialogue["dialogue_id"] in anchors
        ]
        if not anchor_items:
            continue

        speakers_by_key = {
            _speaker_rule_key(speaker): speaker for _, speaker in anchor_items
        }
        if len(speakers_by_key) > 2:
            continue
        if len(speakers_by_key) == 1:
            anchor_speaker = next(iter(speakers_by_key.values()))
            counterpart = _segment_counterpart_speaker(segment, payload, anchor_speaker)
            if not counterpart:
                continue
            speakers_by_key[_speaker_rule_key(counterpart)] = counterpart

        if len(speakers_by_key) != 2:
            continue
        if len(segment) > 6 and len(anchor_items) < 2:
            continue

        ref_index, ref_speaker = anchor_items[0]
        ref_key = _speaker_rule_key(ref_speaker)
        other_key = next(key for key in speakers_by_key if key != ref_key)
        if not _alternation_anchors_are_consistent(
            anchor_items, ref_index, ref_key, other_key
        ):
            continue

        for index, dialogue in enumerate(segment):
            dialogue_id = dialogue["dialogue_id"]
            if dialogue_id in anchors:
                continue
            expected_key = ref_key if (index - ref_index) % 2 == 0 else other_key
            inferred[dialogue_id] = _speaker_with_evidence(
                speakers_by_key[expected_key],
                "规则：相邻两人连续对白按轮次交替",
                ALTERNATION_RULE_CONFIDENCE,
            )
    return inferred


def _standalone_rule_segments(dialogues: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    for dialogue in dialogues:
        if _dialogue_kind(dialogue) != "standalone":
            if current:
                segments.append(current)
                current = []
            continue
        if current and _dialogue_gap_exceeds(
            current[-1], dialogue, ALTERNATION_MAX_PARAGRAPH_GAP
        ):
            segments.append(current)
            current = []
        current.append(dialogue)
    if current:
        segments.append(current)
    return segments


def _alternation_anchors_are_consistent(
    anchor_items: list[tuple[int, dict]],
    ref_index: int,
    ref_key: str,
    other_key: str,
) -> bool:
    for index, speaker in anchor_items:
        expected_key = ref_key if (index - ref_index) % 2 == 0 else other_key
        if _speaker_rule_key(speaker) != expected_key:
            return False
    return True


def _segment_counterpart_speaker(
    segment: list[dict], payload: dict, anchor_speaker: dict
) -> dict | None:
    anchor_key = _speaker_rule_key(anchor_speaker)
    options = [
        option
        for option in _segment_known_options(segment, payload)
        if _speaker_rule_key(option) != anchor_key
    ]
    if len(options) == 1:
        return _speaker_from_option(
            options[0],
            "规则：同一小段只有两个本地说话候选",
            ALTERNATION_RULE_CONFIDENCE,
        )

    npc = _npc_speaker_from_context(_segment_context_texts(segment))
    if npc and _speaker_rule_key(npc) != anchor_key:
        return _speaker_with_evidence(
            npc,
            "规则：同一小段对白发生在村落交易语境",
            ALTERNATION_RULE_CONFIDENCE,
        )
    return None


def _segment_known_options(segment: list[dict], payload: dict) -> list[dict]:
    context_text = "\n".join(_segment_context_texts(segment))
    return [
        option
        for option in _known_speaker_options(payload)
        if _option_mentioned_in_text(option, context_text)
    ]


def _segment_context_texts(segment: list[dict]) -> list[str]:
    texts: list[Any] = []
    for dialogue in segment:
        texts.append(dialogue.get("text", ""))
        texts.append(dialogue.get("paragraph_text", ""))
        texts.append(_outside_quote_text(dialogue))
        texts.extend(
            row.get("text", "")
            for row in _as_list(dialogue.get("prev_context"))
            if isinstance(row, dict)
        )
        texts.extend(
            row.get("text", "")
            for row in _as_list(dialogue.get("next_context"))
            if isinstance(row, dict)
        )
    return [text for text in (_clean_text(value) for value in texts) if text]


def _dialogue_context_text(dialogue: dict) -> str:
    texts: list[Any] = [
        dialogue.get("text", ""),
        dialogue.get("paragraph_text", ""),
        _outside_quote_text(dialogue),
    ]
    texts.extend(
        row.get("text", "")
        for row in _as_list(dialogue.get("prev_context"))
        if isinstance(row, dict)
    )
    texts.extend(
        row.get("text", "")
        for row in _as_list(dialogue.get("next_context"))
        if isinstance(row, dict)
    )
    return "\n".join(text for text in (_clean_text(value) for value in texts) if text)


def _known_speaker_options(payload: dict) -> list[dict]:
    options: list[dict] = []
    for row in _as_list(payload.get("candidate_characters")):
        if not isinstance(row, dict):
            continue
        display = _clean_text(row.get("display_name") or row.get("display"))
        entity_id = _clean_text(row.get("entity_id"))
        if not display or not entity_id:
            continue
        names = _unique_strings([display, *_as_list(row.get("aliases"))], 10)
        names = [name for name in names if _name_appears(name, display + "\n" + name)]
        if not names:
            names = [display]
        options.append(
            {
                "entity_id": entity_id,
                "display": display,
                "status": "known",
                "names": names,
            }
        )
    return options


def _option_mentioned_in_text(option: dict, text: str) -> bool:
    return any(_name_appears(name, text) for name in option.get("names", []))


def _speaker_from_option(option: dict, evidence: str, confidence: float) -> dict:
    return {
        "entity_id": option["entity_id"],
        "display": option["display"],
        "status": option["status"],
        "confidence": confidence,
        "evidence": [evidence],
    }


def _npc_speaker_from_context(texts: list[str]) -> dict | None:
    context_text = "\n".join(_clean_text(text) for text in texts if _clean_text(text))
    if any(marker in context_text for marker in VILLAGE_CONTEXT_MARKERS):
        return {
            "entity_id": f"npc:{_clean_identifier(VILLAGE_NPC_DISPLAY)}",
            "display": VILLAGE_NPC_DISPLAY,
            "status": "npc",
            "confidence": RULE_CONFIDENCE,
            "evidence": ["规则：上下文是深山小村落交易"],
        }
    return None


def _speaker_with_evidence(speaker: dict, evidence: str, confidence: float) -> dict:
    return {
        "entity_id": speaker["entity_id"],
        "display": speaker["display"],
        "status": speaker["status"],
        "confidence": confidence,
        "evidence": _unique_strings([evidence, *_as_list(speaker.get("evidence"))], 4),
    }


def _speaker_rule_key(speaker: dict) -> str:
    entity_id = _clean_text(speaker.get("entity_id") or speaker.get("speaker_entity_id"))
    if entity_id:
        return entity_id
    status = _clean_status(speaker.get("status") or speaker.get("speaker_status"))
    display = _clean_text(speaker.get("display") or speaker.get("speaker_display"))
    return f"{status}:{display}"


def _rule_vote_for_speaker(dialogue: dict, speaker: dict) -> dict:
    parsed_vote = {
        "dialogue_id": dialogue["dialogue_id"],
        "speaker_entity_id": speaker["entity_id"],
        "speaker_display": speaker["display"],
        "speaker_status": speaker["status"],
        "confidence": speaker.get("confidence", RULE_CONFIDENCE),
        "candidate_speakers": [
            {
                "entity_id": speaker["entity_id"],
                "display": speaker["display"],
                "status": speaker["status"],
                "score": speaker.get("confidence", RULE_CONFIDENCE),
            }
        ],
        "evidence": speaker.get("evidence", []),
        "should_create_new_entity": False,
        "new_entity_hint": "",
        "needs_review": False,
    }
    return _normalize_vote(
        parsed_vote=parsed_vote,
        dialogue=dialogue,
        model=RULE_MODEL,
        weight=1.0,
    )


def _aggregate_votes(dialogue: dict, votes: list[dict], config: AnnotationConfig) -> dict:
    base = {
        "dialogue_id": dialogue["dialogue_id"],
        "dialogue_index": dialogue["dialogue_index"],
        "volume_id": dialogue["volume_id"],
        "chapter_id": dialogue["chapter_id"],
        "chapter_index": dialogue["chapter_index"],
        "chapter_title": dialogue["chapter_title"],
        "scene_id": dialogue["scene_id"],
        "scene_index": dialogue["scene_index"],
        "paragraph_id": dialogue["paragraph_id"],
        "paragraph_index": dialogue["paragraph_index"],
        "local_dialogue_index": dialogue["local_dialogue_index"],
        "text": dialogue["text"],
        "quote_text": dialogue["quote_text"],
        "dialogue_kind": _dialogue_kind(dialogue),
        "char_start": dialogue["char_start"],
        "char_end": dialogue["char_end"],
    }
    if not votes:
        return {
            **base,
            "speaker_entity_id": "review:no_vote",
            "speaker_display": REVIEW_LABEL,
            "speaker_status": "review",
            "confidence": 0.0,
            "candidate_speakers": [],
            "evidence": [],
            "needs_review": True,
            "review_reason": "no_valid_votes",
            "vote_count": 0,
            "models": [],
        }

    groups: dict[str, dict] = {}
    total_weight = 0.0
    for vote in votes:
        weight = max(0.0, _safe_float(vote.get("weight"), 1.0))
        total_weight += weight
        key = _vote_group_key(vote)
        group = groups.setdefault(
            key,
            {
                "key": key,
                "speaker_entity_id": vote["speaker_entity_id"],
                "speaker_display": vote["speaker_display"],
                "speaker_status": vote["speaker_status"],
                "score": 0.0,
                "rule_confidence": 0.0,
                "models": set(),
                "evidence": [],
                "needs_review": False,
            },
        )
        group["score"] += weight * _safe_float(vote.get("confidence"), 0.0)
        if _is_rule_vote(vote):
            group["rule_confidence"] = max(
                group["rule_confidence"], _safe_float(vote.get("confidence"), 0.0)
            )
        group["models"].add(vote["model"])
        group["evidence"].extend(vote.get("evidence", []))
        group["needs_review"] = group["needs_review"] or bool(vote.get("needs_review"))

    rule_override_key = _rule_override_key(votes)
    ranked = sorted(groups.values(), key=lambda row: row["score"], reverse=True)
    if rule_override_key and rule_override_key in groups:
        rule_group = groups[rule_override_key]
        ranked = [rule_group] + [
            row for row in ranked if row["key"] != rule_override_key
        ]
    top = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    denominator = total_weight if total_weight > 0 else 1.0
    agreement = top["score"] / denominator
    margin = (top["score"] - second_score) / denominator
    support_count = len(top["models"])
    non_rule_models = {vote["model"] for vote in votes if not _is_rule_vote(vote)}
    participating_model_count = len(non_rule_models) or len({vote["model"] for vote in votes})
    required_support = min(config.min_support_models, max(1, participating_model_count))
    accepted_by_rule = bool(rule_override_key and top["key"] == rule_override_key)
    forced_review_reason = (
        "" if accepted_by_rule else _forced_review_reason(dialogue, top, participating_model_count)
    )

    if accepted_by_rule:
        accepted = (
            not top["needs_review"]
            and top["speaker_status"] in {"known", "npc"}
            and top["rule_confidence"] >= ALTERNATION_RULE_CONFIDENCE
        )
    elif forced_review_reason:
        accepted = False
    elif participating_model_count <= 1:
        accepted = (
            not top["needs_review"]
            and top["speaker_status"] in {"known", "mystery", "npc"}
            and agreement >= config.min_confidence
        )
    else:
        accepted = (
            not top["needs_review"]
            and top["speaker_status"] in {"known", "mystery", "npc"}
            and agreement >= config.min_agreement
            and margin >= config.min_margin
            and support_count >= required_support
        )

    candidate_speakers = [
        {
            "entity_id": row["speaker_entity_id"],
            "display": row["speaker_display"],
            "status": row["speaker_status"],
            "score": round(row["score"] / denominator, 4),
            "models": sorted(row["models"]),
        }
        for row in ranked
    ]

    return {
        **base,
        "speaker_entity_id": top["speaker_entity_id"],
        "speaker_display": top["speaker_display"] or REVIEW_LABEL,
        "speaker_status": top["speaker_status"] if accepted else "review",
        "confidence": round(
            max(agreement, top["rule_confidence"]) if accepted_by_rule else agreement,
            4,
        ),
        "candidate_speakers": candidate_speakers,
        "evidence": _unique_strings(top["evidence"], 8),
        "needs_review": not accepted,
        "rule_applied": accepted_by_rule and accepted,
        "review_reason": ""
        if accepted
        else _review_reason(
            top,
            agreement,
            margin,
            support_count,
            forced_review_reason=forced_review_reason,
        ),
        "vote_count": len(votes),
        "models": sorted({vote["model"] for vote in votes}),
    }


def _forced_review_reason(
    dialogue: dict, top: dict, participating_model_count: int
) -> str:
    if _dialogue_kind(dialogue) != "standalone":
        return "non_standalone_quote"
    if participating_model_count > 1:
        return ""
    status = _clean_status(top.get("speaker_status"))
    if status == "mystery":
        return "single_model_mystery_candidate"
    if status == "known" and not _has_speaker_anchor(dialogue, top):
        return "single_model_without_anchor"
    return ""


def _has_speaker_anchor(dialogue: dict, top: dict) -> bool:
    display = _clean_text(top.get("speaker_display"))
    if not display:
        return False
    if _self_intro_mentions(dialogue.get("text", ""), display):
        return True
    return _context_attributes_speech_to(
        dialogue=dialogue,
        display=display,
    )


def _self_intro_mentions(text: Any, display: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned or display not in cleaned:
        return False
    text_until_display = cleaned[: cleaned.find(display) + len(display)]
    return any(marker in text_until_display for marker in SELF_INTRO_MARKERS)


def _context_attributes_speech_to(dialogue: dict, display: str) -> bool:
    for text in _direct_attribution_contexts(dialogue):
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        if _speech_verb_near_name(cleaned, display):
            return True
    return False


def _speech_verb_near_name(text: str, display: str) -> bool:
    if display not in text:
        return False
    for verb in SPEECH_VERBS:
        name_then_verb = f"{re.escape(display)}.{{0,30}}{re.escape(verb)}"
        for match in re.finditer(name_then_verb, text):
            if not _speech_verb_match_is_indirect(text, match.end(), verb):
                return True
    return False


def _speech_verb_match_is_indirect(text: str, match_end: int, verb: str) -> bool:
    verb_start = match_end - len(verb)
    if verb_start > 0 and text[verb_start - 1] in {"被", "询"}:
        return True
    if verb not in {"说", "问"}:
        return False
    suffix = text[match_end : match_end + 4]
    return any(suffix.startswith(value) for value in INDIRECT_SPEECH_SUFFIXES)


def _is_rule_vote(vote: dict) -> bool:
    return _clean_text(vote.get("model")).startswith(RULE_MODEL_PREFIX)


def _rule_override_key(votes: list[dict]) -> str:
    keys = {
        _vote_group_key(vote)
        for vote in votes
        if _is_rule_vote(vote)
        and not vote.get("needs_review")
        and _clean_status(vote.get("speaker_status")) in {"known", "npc"}
        and _safe_float(vote.get("confidence"), 0.0) >= ALTERNATION_RULE_CONFIDENCE
    }
    if len(keys) == 1:
        return next(iter(keys))
    return ""


def _vote_group_key(vote: dict) -> str:
    entity_id = _clean_text(vote.get("speaker_entity_id"))
    if entity_id:
        return entity_id
    status = _clean_status(vote.get("speaker_status"))
    display = _clean_text(vote.get("speaker_display")) or "unknown"
    return f"{status}:{display}"


def _review_reason(
    top: dict,
    agreement: float,
    margin: float,
    support_count: int,
    forced_review_reason: str = "",
) -> str:
    if forced_review_reason:
        return forced_review_reason
    if top["needs_review"]:
        return "model_requested_review"
    if top["speaker_status"] not in {"known", "mystery", "npc"}:
        return f"non_final_status:{top['speaker_status']}"
    if agreement <= 0:
        return "zero_confidence"
    return f"low_agreement:{agreement:.2f};margin:{margin:.2f};support:{support_count}"


def _write_labeled_text(
    paragraphs: list[dict], annotations: list[dict], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_labeled_text(paragraphs, annotations), encoding="utf-8", newline="\n"
    )


def _render_labeled_text(paragraphs: list[dict], annotations: list[dict]) -> str:
    annotations_by_paragraph: dict[str, list[dict]] = {}
    for annotation in annotations:
        annotations_by_paragraph.setdefault(annotation["paragraph_id"], []).append(annotation)

    rendered: list[str] = []
    for paragraph in paragraphs:
        text = paragraph["text"]
        paragraph_annotations = sorted(
            annotations_by_paragraph.get(paragraph["paragraph_id"], []),
            key=lambda row: int(row.get("char_start", 0)),
            reverse=True,
        )
        for annotation in paragraph_annotations:
            label = _label_for_annotation(annotation)
            if not label:
                continue
            insert_at = int(annotation["char_start"])
            text = f"{text[:insert_at]}【{label}】{text[insert_at:]}"
        rendered.append(text)
    return "\n".join(rendered) + "\n"


def _label_for_annotation(annotation: dict) -> str:
    if _dialogue_kind(annotation) != "standalone":
        return ""
    if annotation.get("needs_review"):
        return REVIEW_LABEL
    status = _clean_status(annotation.get("speaker_status"))
    display = _clean_text(annotation.get("speaker_display"))
    if status == "npc" and not display:
        return "路人"
    if status in {"known", "mystery", "npc"} and display:
        return display
    return REVIEW_LABEL


def _clean_status(value: Any) -> str:
    status = _clean_text(value)
    if status in SPEAKER_STATUSES:
        return status
    return "review"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_identifier(value: Any) -> str:
    text = _clean_text(value)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text, flags=re.U)
    return text.strip("_") or "unknown"


def _safe_model_name(model: str) -> str:
    return _clean_identifier(model.replace(":", "_").replace("/", "_"))


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_strings(values: list[Any], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
            if limit is not None and len(result) >= limit:
                break
    return result


def _truncate(value: Any, max_length: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 1)] + "…"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _output_path_for_log(path: Path) -> str:
    return str(path).replace("\\", "/")
