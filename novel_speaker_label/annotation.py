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
    start_dialogue_index: int = 0
    dialogue_limit: int | None = None
    min_confidence: float = 0.55
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

    characters = _read_optional_jsonl(memory_dir / "semantic" / "characters.jsonl")
    mysteries = _read_optional_jsonl(memory_dir / "mystery_entities.jsonl")
    scene_memories = _read_optional_jsonl(memory_dir / "episodic" / "scenes.jsonl")
    scene_memories_by_parent = _group_scene_memories(scene_memories)

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

    for request_number, dialogue in enumerate(dialogues, start=1):
        print(
            "[annotate] "
            f"{request_number}/{len(dialogues)} {dialogue['dialogue_id']} "
            f"scene={dialogue['scene_id']}",
            flush=True,
        )
        payload = _build_annotation_payload(
            dialogue=dialogue,
            paragraphs_by_id=paragraphs_by_id,
            scene_memories_by_parent=scene_memories_by_parent,
            characters=characters,
            mysteries=mysteries,
            volume_meta=volume_meta,
            config=config,
        )
        dialogue_votes: list[dict] = []

        for model in config.models:
            request_id = f"{dialogue['dialogue_id']}--{_safe_model_name(model)}"
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
                    parsed_vote = read_json(cached_path)
                    response_text = json.dumps(parsed_vote, ensure_ascii=False)
                elif config.cache_only:
                    raise FileNotFoundError(f"Cached annotation vote not found: {cached_path}")
                else:
                    if not prompt:
                        prompt = _build_annotation_prompt(payload)
                    response_text = clients[model].generate(prompt)
                    (raw_dir / f"{request_id}.txt").write_text(
                        response_text, encoding="utf-8", newline="\n"
                    )
                    parsed_vote = _parse_json_response(response_text)
                    write_json(cached_path, parsed_vote)
            except Exception as exc:
                failure = {
                    "request_id": request_id,
                    "dialogue_id": dialogue["dialogue_id"],
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

            vote = _normalize_vote(
                parsed_vote=parsed_vote,
                dialogue=dialogue,
                model=model,
                weight=config.model_weights.get(model, 1.0),
            )
            votes.append(vote)
            dialogue_votes.append(vote)

        if not config.dry_run:
            annotations.append(_aggregate_votes(dialogue, dialogue_votes, config))

    if config.dry_run:
        write_json(
            annotation_dir / "dry_run.json",
            {
                "volume_id": volume_meta["volume_id"],
                "models": list(config.models),
                "dialogue_count": len(dialogues),
                "prompt_count": prompt_count,
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
        "prompt_count": prompt_count,
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
    dialogue: dict,
    paragraphs_by_id: dict[str, dict],
    scene_memories_by_parent: dict[str, list[dict]],
    characters: list[dict],
    mysteries: list[dict],
    volume_meta: dict,
    config: AnnotationConfig,
) -> dict:
    current_paragraph = paragraphs_by_id.get(dialogue["paragraph_id"], {})
    context_rows = _dialogue_context_rows(dialogue, current_paragraph)
    scene_memories = scene_memories_by_parent.get(dialogue["scene_id"], [])[
        : config.max_scene_summaries
    ]
    active_names = _active_names(scene_memories)
    context_text = _context_text(dialogue, context_rows, scene_memories)
    context_ids = {row.get("paragraph_id") for row in context_rows if row.get("paragraph_id")}
    context_ids.add(dialogue["dialogue_id"])

    return {
        "volume": {
            "volume_id": volume_meta["volume_id"],
            "volume": volume_meta.get("volume"),
        },
        "dialogue": {
            "dialogue_id": dialogue["dialogue_id"],
            "dialogue_index": dialogue["dialogue_index"],
            "chapter_id": dialogue["chapter_id"],
            "chapter_title": dialogue["chapter_title"],
            "scene_id": dialogue["scene_id"],
            "paragraph_id": dialogue["paragraph_id"],
            "text": dialogue["text"],
            "quote_text": dialogue["quote_text"],
            "paragraph_text": dialogue["paragraph_text"],
        },
        "context": {
            "previous_paragraphs": _context_from_dialogue(dialogue, "prev_context"),
            "current_paragraph": {
                "paragraph_id": current_paragraph.get("paragraph_id"),
                "text": current_paragraph.get("text", dialogue.get("paragraph_text", "")),
            },
            "next_paragraphs": _context_from_dialogue(dialogue, "next_context"),
        },
        "scene_memory": [_scene_card(row) for row in scene_memories],
        "candidate_characters": [
            _character_card(row)
            for row in _select_character_candidates(
                characters=characters,
                active_names=active_names,
                context_text=context_text,
                scene_id=dialogue["scene_id"],
                limit=config.max_characters,
            )
        ],
        "candidate_mysteries": [
            _mystery_card(row)
            for row in _select_mystery_candidates(
                mysteries=mysteries,
                context_ids=context_ids,
                context_text=context_text,
                scene_id=dialogue["scene_id"],
                limit=config.max_mysteries,
            )
        ],
    }


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


def _select_character_candidates(
    characters: list[dict],
    active_names: set[str],
    context_text: str,
    scene_id: str,
    limit: int,
) -> list[dict]:
    scored = [
        (_score_character(row, active_names, context_text, scene_id), index, row)
        for index, row in enumerate(characters)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[: max(1, limit)]]


def _score_character(
    character: dict, active_names: set[str], context_text: str, scene_id: str
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
        if _name_appears(_clean_text(title), context_text):
            score += 2.0
    if str(character.get("first_seen_scene_id", "")).startswith(scene_id):
        score += 1.0
    if str(character.get("latest_seen_scene_id", "")).startswith(scene_id):
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
    limit: int,
) -> list[dict]:
    scored = [
        (_score_mystery(row, context_ids, context_text, scene_id), index, row)
        for index, row in enumerate(mysteries)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[: max(1, limit)]]


def _score_mystery(
    mystery: dict, context_ids: set[str], context_text: str, scene_id: str
) -> float:
    score = _safe_float(mystery.get("confidence"), 0.5) * 0.1
    if str(mystery.get("source_scene_id", "")).startswith(scene_id):
        score += 6.0
    for evidence in _as_list(mystery.get("evidence")):
        if evidence in context_ids:
            score += 8.0
    temporary_name = _clean_text(mystery.get("temporary_name"))
    if _name_appears(temporary_name, context_text):
        score += 4.0
    return score


def _scene_card(row: dict) -> dict:
    return {
        "scene_id": row.get("scene_id"),
        "parent_scene_id": row.get("parent_scene_id"),
        "chunk_index": row.get("chunk_index"),
        "chunk_count": row.get("chunk_count"),
        "scene_summary": _truncate(row.get("scene_summary", ""), 500),
        "active_characters": _unique_strings(_as_list(row.get("active_characters")), 20),
        "relationships": _unique_strings(_as_list(row.get("relationships")), 12),
        "notes": _truncate(row.get("notes", ""), 240),
    }


def _character_card(row: dict) -> dict:
    return {
        "entity_id": row.get("entity_id"),
        "display_name": row.get("display_name"),
        "aliases": _unique_strings(_as_list(row.get("aliases")), 12),
        "titles": _unique_strings(_as_list(row.get("titles")), 12),
        "description": _truncate(row.get("description", ""), 700),
        "speech_style": _truncate(row.get("speech_style", ""), 240),
        "relationship_hints": _unique_strings(
            _as_list(row.get("relationship_hints")), 12
        ),
        "first_seen_scene_id": row.get("first_seen_scene_id"),
        "latest_seen_scene_id": row.get("latest_seen_scene_id"),
        "confidence": _safe_float(row.get("confidence"), 0.5),
    }


def _mystery_card(row: dict) -> dict:
    return {
        "mystery_id": _mystery_id(row),
        "temporary_name": row.get("temporary_name"),
        "description": _truncate(row.get("description", ""), 400),
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
        "你的任务只判断输入 dialogue 这一句台词的说话人，不要改写原文。\n\n"
        "规则：\n"
        "1. 只输出严格 JSON，不要 Markdown，不要解释性前后缀。\n"
        "2. 如果说话人是候选角色，speaker_entity_id 必须使用 candidate_characters 里的 entity_id，speaker_status 填 known。\n"
        "3. 如果说话人像某个未知人物，speaker_entity_id 使用 candidate_mysteries 里的 mystery_id，speaker_status 填 mystery。\n"
        "4. 如果只是无法追踪的路人/群体，speaker_status 填 npc，speaker_entity_id 可为空或使用 npc:简短称呼。\n"
        "5. 证据不足时不要硬猜，speaker_status 填 ambiguous 或 review，并把 needs_review 设为 true。\n"
        "6. candidate_speakers 至少列出 1 个候选；如果无法判断，列出最可能的候选和不确定原因。\n"
        "7. confidence 使用 0 到 1 之间的小数，表示你对最终判断的把握。\n\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "dialogue_id": "string",\n'
        '  "speaker_entity_id": "string",\n'
        '  "speaker_display": "string",\n'
        '  "speaker_status": "known|mystery|npc|ambiguous|review",\n'
        '  "confidence": 0.0,\n'
        '  "candidate_speakers": [\n'
        '    {"entity_id": "string", "display": "string", "status": "known|mystery|npc|ambiguous|review", "score": 0.0}\n'
        "  ],\n"
        '  "evidence": ["短证据，不要大段复制原文"],\n'
        '  "should_create_new_entity": false,\n'
        '  "new_entity_hint": "",\n'
        '  "needs_review": false\n'
        "}\n\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _parse_json_response(response_text: str) -> dict:
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
                "speaker_entity_id": vote["speaker_entity_id"],
                "speaker_display": vote["speaker_display"],
                "speaker_status": vote["speaker_status"],
                "score": 0.0,
                "models": set(),
                "evidence": [],
                "needs_review": False,
            },
        )
        group["score"] += weight * _safe_float(vote.get("confidence"), 0.0)
        group["models"].add(vote["model"])
        group["evidence"].extend(vote.get("evidence", []))
        group["needs_review"] = group["needs_review"] or bool(vote.get("needs_review"))

    ranked = sorted(groups.values(), key=lambda row: row["score"], reverse=True)
    top = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    denominator = total_weight if total_weight > 0 else 1.0
    agreement = top["score"] / denominator
    margin = (top["score"] - second_score) / denominator
    support_count = len(top["models"])
    participating_model_count = len({vote["model"] for vote in votes})
    required_support = min(config.min_support_models, max(1, participating_model_count))

    if participating_model_count <= 1:
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
        "confidence": round(agreement, 4),
        "candidate_speakers": candidate_speakers,
        "evidence": _unique_strings(top["evidence"], 8),
        "needs_review": not accepted,
        "review_reason": "" if accepted else _review_reason(top, agreement, margin, support_count),
        "vote_count": len(votes),
        "models": sorted({vote["model"] for vote in votes}),
    }


def _vote_group_key(vote: dict) -> str:
    entity_id = _clean_text(vote.get("speaker_entity_id"))
    if entity_id:
        return entity_id
    status = _clean_status(vote.get("speaker_status"))
    display = _clean_text(vote.get("speaker_display")) or "unknown"
    return f"{status}:{display}"


def _review_reason(
    top: dict, agreement: float, margin: float, support_count: int
) -> str:
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
