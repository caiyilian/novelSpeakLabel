from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .jsonl import read_json, read_jsonl, write_json, write_jsonl
from .ollama_client import OllamaClient, OllamaConfig

READING_V2_TASKS = (
    "annotation",
    "chunk_summary",
    "entity_discovery",
    "entity_update",
    "global_summary",
    "repair",
)
SPEAKER_STATUSES = {"known", "mystery", "npc_group", "unknown", "review"}
IMPORTANCE_LEVELS = {"minor": 1, "medium": 2, "major": 3}
REVIEW_LABEL = "待复核"


@dataclass(frozen=True)
class ReadingV2Config:
    output_dir: Path
    model: str = "qwen3:32b"
    ollama_host: str = "http://127.0.0.1:11434"
    timeout: int = 1800
    temperature: float = 0.0
    num_predict: int = 4096
    dry_run: bool = False
    overwrite_cache: bool = False
    cache_only: bool = False
    continue_on_error: bool = True
    write_prompts: bool = True
    max_paragraphs_per_chunk: int = 16
    max_dialogues_per_chunk: int = 32
    lookback_paragraphs: int = 3
    lookahead_paragraphs: int = 1
    start_chunk_index: int = 0
    chunk_limit: int | None = None
    max_prompt_tokens: int = 24000
    hard_prompt_tokens: int = 30000
    max_candidate_entities: int = 12


@dataclass
class ReadingV2State:
    global_summary: str = ""
    chunk_summaries: list[dict] = field(default_factory=list)
    characters: dict[str, dict] = field(default_factory=dict)
    mysteries: dict[str, dict] = field(default_factory=dict)
    npc_groups: dict[str, dict] = field(default_factory=dict)
    facts: list[dict] = field(default_factory=list)
    entity_events: list[dict] = field(default_factory=list)
    next_character_id: int = 1
    next_mystery_id: int = 1
    next_npc_group_id: int = 1


def annotate_v2_volume(config: ReadingV2Config) -> dict:
    volume_meta = read_json(config.output_dir / "volume.json")
    preprocess_dir = config.output_dir / "preprocess"
    paragraphs = _read_required_jsonl(preprocess_dir / "paragraphs.jsonl")
    dialogues = _read_optional_jsonl(preprocess_dir / "dialogues.jsonl")
    chunks = _build_reading_chunks(
        paragraphs=paragraphs,
        dialogues=dialogues,
        volume_id=volume_meta["volume_id"],
        config=config,
    )
    selected_chunks = _select_chunks(chunks, config)

    reading_dir = config.output_dir / "reading_v2"
    memory_dir = config.output_dir / "memory_v2"
    annotation_dir = config.output_dir / "annotation_v2"
    prompt_dir = reading_dir / "prompts"
    raw_dir = reading_dir / "raw"
    cache_dir = reading_dir / "cache"
    failure_dir = reading_dir / "failures"
    for path in (
        reading_dir,
        memory_dir,
        annotation_dir,
        prompt_dir,
        raw_dir,
        cache_dir,
        failure_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    write_jsonl(reading_dir / "chunks.jsonl", selected_chunks)

    client = OllamaClient(
        OllamaConfig(
            host=config.ollama_host,
            model=config.model,
            timeout=config.timeout,
            temperature=config.temperature,
            num_predict=config.num_predict,
        )
    )
    state = ReadingV2State()
    annotations_by_id: dict[str, dict] = {}
    repairs: list[dict] = []
    failed_requests: list[dict] = []
    prompt_reports: list[dict] = []

    for offset, chunk in enumerate(selected_chunks, start=1):
        print(
            "[annotate-v2] "
            f"{offset}/{len(selected_chunks)} {chunk['chunk_id']} "
            f"paragraphs={len(chunk['paragraphs'])} dialogues={len(chunk['dialogues'])}",
            flush=True,
        )
        payload = _build_base_payload(
            volume_meta=volume_meta,
            chunk=chunk,
            state=state,
            config=config,
        )

        annotation_response = _run_reading_task(
            task="annotation",
            chunk=chunk,
            payload=payload,
            prompt_builder=_build_annotation_prompt,
            client=client,
            config=config,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            failure_dir=failure_dir,
            failed_requests=failed_requests,
            prompt_reports=prompt_reports,
        )
        chunk_annotations = _extract_annotations(
            annotation_response,
            chunk=chunk,
            request_id=_task_request_id(chunk["chunk_id"], "annotation", config.model),
            dry_run=config.dry_run,
        )
        _ensure_entities_from_annotations(chunk_annotations, state, chunk["chunk_id"])

        summary_payload = {
            **payload,
            "task_a_annotations": chunk_annotations,
        }
        chunk_summary_response = _run_reading_task(
            task="chunk_summary",
            chunk=chunk,
            payload=summary_payload,
            prompt_builder=_build_chunk_summary_prompt,
            client=client,
            config=config,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            failure_dir=failure_dir,
            failed_requests=failed_requests,
            prompt_reports=prompt_reports,
        )
        chunk_summary = _normalize_chunk_summary(
            chunk_summary_response, chunk=chunk, dry_run=config.dry_run
        )
        if chunk_summary:
            state.chunk_summaries.append(chunk_summary)

        discovery_payload = {
            **payload,
            "task_a_annotations": chunk_annotations,
            "task_b_chunk_summary": chunk_summary,
        }
        entity_discovery_response = _run_reading_task(
            task="entity_discovery",
            chunk=chunk,
            payload=discovery_payload,
            prompt_builder=_build_entity_discovery_prompt,
            client=client,
            config=config,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            failure_dir=failure_dir,
            failed_requests=failed_requests,
            prompt_reports=prompt_reports,
        )
        discovery_events = _apply_entity_discovery(
            entity_discovery_response, state=state, chunk_id=chunk["chunk_id"]
        )

        update_payload = {
            **payload,
            "task_a_annotations": chunk_annotations,
            "task_b_chunk_summary": chunk_summary,
            "task_d_entity_discovery": entity_discovery_response or {},
            "current_entity_memory": _memory_snapshot_for_output(state),
        }
        entity_update_response = _run_reading_task(
            task="entity_update",
            chunk=chunk,
            payload=update_payload,
            prompt_builder=_build_entity_update_prompt,
            client=client,
            config=config,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            failure_dir=failure_dir,
            failed_requests=failed_requests,
            prompt_reports=prompt_reports,
        )
        update_events = _apply_entity_updates(
            entity_update_response, state=state, chunk_id=chunk["chunk_id"]
        )

        global_payload = {
            **payload,
            "task_b_chunk_summary": chunk_summary,
            "task_d_entity_discovery": entity_discovery_response or {},
            "task_e_entity_update": entity_update_response or {},
            "old_global_summary": state.global_summary,
        }
        global_summary_response = _run_reading_task(
            task="global_summary",
            chunk=chunk,
            payload=global_payload,
            prompt_builder=_build_global_summary_prompt,
            client=client,
            config=config,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            failure_dir=failure_dir,
            failed_requests=failed_requests,
            prompt_reports=prompt_reports,
        )
        _apply_global_summary(
            global_summary_response, state=state, chunk_id=chunk["chunk_id"]
        )

        repair_payload = {
            **payload,
            "task_a_annotations": chunk_annotations,
            "task_b_chunk_summary": chunk_summary,
            "updated_global_summary": state.global_summary,
            "updated_entity_memory": _memory_snapshot_for_output(state),
            "unresolved_annotations": [
                row
                for row in chunk_annotations
                if row.get("needs_review")
                or row.get("speaker_status") in {"unknown", "review", "mystery"}
            ],
        }
        repair_response = _run_reading_task(
            task="repair",
            chunk=chunk,
            payload=repair_payload,
            prompt_builder=_build_repair_prompt,
            client=client,
            config=config,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            failure_dir=failure_dir,
            failed_requests=failed_requests,
            prompt_reports=prompt_reports,
        )
        chunk_repairs = _extract_repairs(
            repair_response,
            chunk=chunk,
            annotations=chunk_annotations,
            request_id=_task_request_id(chunk["chunk_id"], "repair", config.model),
        )
        repairs.extend(chunk_repairs)
        chunk_annotations = _apply_repairs(chunk_annotations, chunk_repairs)

        if not config.dry_run:
            chunk_annotations = _fill_missing_annotations(
                chunk=chunk,
                annotations=chunk_annotations,
            )
            for annotation in chunk_annotations:
                annotations_by_id[annotation["dialogue_id"]] = annotation
            _append_entity_events(
                state=state,
                chunk_id=chunk["chunk_id"],
                events=[*discovery_events, *update_events],
            )

    annotations = sorted(
        annotations_by_id.values(), key=lambda row: int(row.get("dialogue_index", 0))
    )
    prompt_summary = _write_prompt_length_reports(
        reading_dir=reading_dir,
        reports=prompt_reports,
    )
    _write_memory_outputs(memory_dir=memory_dir, state=state, volume_meta=volume_meta)
    _write_annotation_outputs(
        annotation_dir=annotation_dir,
        paragraphs=paragraphs,
        annotations=annotations,
        repairs=repairs,
        dry_run=config.dry_run,
    )
    write_jsonl(reading_dir / "failed_requests.jsonl", failed_requests)

    review_count = sum(1 for row in annotations if row.get("needs_review"))
    summary = {
        "volume_id": volume_meta["volume_id"],
        "model": config.model,
        "pipeline": "reading-v2",
        "output_dir": _output_path_for_log(config.output_dir),
        "dry_run": config.dry_run,
        "cache_only": config.cache_only,
        "chunk_count": len(selected_chunks),
        "dialogue_count": sum(len(chunk["dialogues"]) for chunk in selected_chunks),
        "annotation_count": len(annotations),
        "repair_count": len(repairs),
        "review_count": review_count,
        "prompt_count": len(prompt_reports),
        "max_prompt_estimated_tokens": prompt_summary["max_estimated_tokens"],
        "avg_prompt_estimated_tokens": prompt_summary["avg_estimated_tokens"],
        "prompt_near_limit_count": prompt_summary["near_limit_count"],
        "prompt_over_limit_count": prompt_summary["over_limit_count"],
        "failed_request_count": len(failed_requests),
    }
    write_json(reading_dir / "run_summary.json", summary)
    write_json(annotation_dir / "run_summary.json", summary)
    print(
        "[annotate-v2] wrote "
        f"{_output_path_for_log(reading_dir / 'prompt_length_report.json')} "
        f"prompts={summary['prompt_count']} "
        f"over_limit={summary['prompt_over_limit_count']} "
        f"failures={summary['failed_request_count']}",
        flush=True,
    )
    return summary


def estimate_prompt_tokens(prompt: str) -> int:
    score = 0.0
    for char in prompt:
        if "\u4e00" <= char <= "\u9fff":
            score += 1.6
        elif char.isascii() and char.isalnum():
            score += 0.4
        elif char.isspace():
            score += 0.2
        else:
            score += 0.6
    return int(math.ceil(score))


def prompt_length_status(
    estimated_tokens: int, target_token_limit: int, hard_token_limit: int
) -> str:
    if estimated_tokens > hard_token_limit:
        return "over_limit"
    if estimated_tokens > target_token_limit:
        return "near_limit"
    return "ok"


def _run_reading_task(
    *,
    task: str,
    chunk: dict,
    payload: dict,
    prompt_builder: Callable[[dict], str],
    client: OllamaClient,
    config: ReadingV2Config,
    prompt_dir: Path,
    raw_dir: Path,
    cache_dir: Path,
    failure_dir: Path,
    failed_requests: list[dict],
    prompt_reports: list[dict],
) -> Any:
    request_id = _task_request_id(chunk["chunk_id"], task, config.model)
    prompt = prompt_builder(payload)
    prompt_path = prompt_dir / f"{request_id}.txt"
    if config.write_prompts or config.dry_run:
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
    report = _prompt_length_report(
        chunk=chunk,
        task=task,
        prompt=prompt,
        prompt_path=prompt_path,
        output_dir=config.output_dir,
        config=config,
    )
    prompt_reports.append(report)
    if config.dry_run:
        return None
    if report["status"] == "over_limit":
        exc = RuntimeError(
            f"Prompt over hard limit: estimated={report['estimated_tokens']} "
            f"hard={report['hard_token_limit']}"
        )
        _record_failure(
            request_id=request_id,
            chunk=chunk,
            task=task,
            model=config.model,
            prompt_path=prompt_path,
            exc=exc,
            failed_requests=failed_requests,
            failure_dir=failure_dir,
            config=config,
        )
        return None

    cached_path = cache_dir / f"{request_id}.json"
    try:
        if cached_path.exists() and not config.overwrite_cache:
            return read_json(cached_path)
        if config.cache_only:
            raise FileNotFoundError(f"Cached reading-v2 response not found: {cached_path}")
        response_text = client.generate(prompt)
        (raw_dir / f"{request_id}.txt").write_text(
            response_text, encoding="utf-8", newline="\n"
        )
        parsed_response = _parse_json_response(response_text)
        write_json(cached_path, parsed_response)
        return parsed_response
    except Exception as exc:
        _record_failure(
            request_id=request_id,
            chunk=chunk,
            task=task,
            model=config.model,
            prompt_path=prompt_path,
            exc=exc,
            failed_requests=failed_requests,
            failure_dir=failure_dir,
            config=config,
        )
        return None


def _record_failure(
    *,
    request_id: str,
    chunk: dict,
    task: str,
    model: str,
    prompt_path: Path,
    exc: Exception,
    failed_requests: list[dict],
    failure_dir: Path,
    config: ReadingV2Config,
) -> None:
    failure = {
        "request_id": request_id,
        "chunk_id": chunk["chunk_id"],
        "task": task,
        "model": model,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "prompt_path": _output_path_for_log(prompt_path),
    }
    write_json(failure_dir / f"{request_id}.json", failure)
    failed_requests.append(failure)
    print(
        "[annotate-v2] failed "
        f"{request_id}: {failure['error_type']}: {failure['error']}",
        flush=True,
    )
    if not config.continue_on_error:
        raise RuntimeError(
            f"reading-v2 task failed: {request_id}. Prompt is saved at {prompt_path}"
        ) from exc


def _build_reading_chunks(
    *,
    paragraphs: list[dict],
    dialogues: list[dict],
    volume_id: str,
    config: ReadingV2Config,
) -> list[dict]:
    sorted_paragraphs = sorted(
        paragraphs, key=lambda row: int(row.get("paragraph_index", 0))
    )
    paragraph_position = {
        paragraph["paragraph_id"]: index for index, paragraph in enumerate(sorted_paragraphs)
    }
    dialogues_by_paragraph: dict[str, list[dict]] = {}
    for dialogue in sorted(dialogues, key=lambda row: int(row.get("dialogue_index", 0))):
        dialogues_by_paragraph.setdefault(dialogue["paragraph_id"], []).append(dialogue)

    chunks: list[dict] = []
    current_paragraphs: list[dict] = []
    current_dialogues: list[dict] = []
    current_scene_id = ""
    max_paragraphs = max(1, config.max_paragraphs_per_chunk)
    max_dialogues = max(1, config.max_dialogues_per_chunk)

    def flush() -> None:
        if not current_paragraphs:
            return
        chunk_index = len(chunks)
        start_position = paragraph_position[current_paragraphs[0]["paragraph_id"]]
        end_position = paragraph_position[current_paragraphs[-1]["paragraph_id"]]
        chunk_id = f"{volume_id}-r{chunk_index + 1:06d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "volume_id": volume_id,
                "chapter_id": _first_text(current_paragraphs, "chapter_id")
                or _first_text(current_dialogues, "chapter_id"),
                "chapter_index": _first_int(current_paragraphs, "chapter_index")
                or _first_int(current_dialogues, "chapter_index"),
                "chapter_title": _first_text(current_paragraphs, "chapter_title")
                or _first_text(current_dialogues, "chapter_title"),
                "scene_id": _first_text(current_paragraphs, "scene_id")
                or _first_text(current_dialogues, "scene_id"),
                "scene_index": _first_int(current_paragraphs, "scene_index")
                or _first_int(current_dialogues, "scene_index"),
                "start_paragraph_index": int(
                    current_paragraphs[0].get("paragraph_index", start_position)
                ),
                "end_paragraph_index": int(
                    current_paragraphs[-1].get("paragraph_index", end_position)
                ),
                "paragraphs": [_paragraph_card(row) for row in current_paragraphs],
                "dialogues": [_dialogue_card(row) for row in current_dialogues],
                "lookback_paragraphs": [
                    _paragraph_card(row)
                    for row in sorted_paragraphs[
                        max(0, start_position - max(0, config.lookback_paragraphs)) :
                        start_position
                    ]
                ],
                "lookahead_paragraphs": [
                    _paragraph_card(row)
                    for row in sorted_paragraphs[
                        end_position
                        + 1 : min(
                            len(sorted_paragraphs),
                            end_position + 1 + max(0, config.lookahead_paragraphs),
                        )
                    ]
                ],
            }
        )

    for paragraph in sorted_paragraphs:
        paragraph_id = paragraph["paragraph_id"]
        paragraph_dialogues = dialogues_by_paragraph.get(paragraph_id, [])
        scene_id = (
            _clean_text(paragraph.get("scene_id"))
            or _first_text(paragraph_dialogues, "scene_id")
            or ""
        )
        chapter_id = (
            _clean_text(paragraph.get("chapter_id"))
            or _first_text(paragraph_dialogues, "chapter_id")
            or ""
        )
        current_chapter_id = (
            _first_text(current_paragraphs, "chapter_id")
            or _first_text(current_dialogues, "chapter_id")
            or ""
        )
        crosses_boundary = bool(
            current_paragraphs
            and (
                (current_scene_id and scene_id and scene_id != current_scene_id)
                or (current_chapter_id and chapter_id and chapter_id != current_chapter_id)
            )
        )
        exceeds_size = bool(
            current_paragraphs
            and (
                len(current_paragraphs) + 1 > max_paragraphs
                or len(current_dialogues) + len(paragraph_dialogues) > max_dialogues
            )
        )
        if crosses_boundary or exceeds_size:
            flush()
            current_paragraphs = []
            current_dialogues = []
            current_scene_id = ""

        current_paragraphs.append(paragraph)
        current_dialogues.extend(paragraph_dialogues)
        current_scene_id = current_scene_id or scene_id

    flush()
    return chunks


def _select_chunks(chunks: list[dict], config: ReadingV2Config) -> list[dict]:
    start = max(0, config.start_chunk_index)
    selected = [chunk for chunk in chunks if chunk["chunk_index"] >= start]
    if config.chunk_limit is not None:
        selected = selected[: max(0, config.chunk_limit)]
    return selected


def _build_base_payload(
    *,
    volume_meta: dict,
    chunk: dict,
    state: ReadingV2State,
    config: ReadingV2Config,
) -> dict:
    return {
        "volume": {
            "volume_id": volume_meta.get("volume_id"),
            "volume": volume_meta.get("volume"),
            "title": volume_meta.get("title", ""),
        },
        "chunk": {
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "chapter_id": chunk.get("chapter_id", ""),
            "chapter_title": chunk.get("chapter_title", ""),
            "scene_id": chunk.get("scene_id", ""),
            "paragraphs": chunk["paragraphs"],
            "dialogues": chunk["dialogues"],
        },
        "near_context": {
            "lookback_paragraphs": chunk["lookback_paragraphs"],
            "lookahead_paragraphs": chunk["lookahead_paragraphs"],
        },
        "memory": {
            "previous_chunk_summary": state.chunk_summaries[-1]
            if state.chunk_summaries
            else {},
            "global_summary": state.global_summary,
            "candidate_entities": _candidate_entity_cards(state, config),
        },
        "rules": {
            "speaker_statuses": sorted(SPEAKER_STATUSES),
            "importance_levels": sorted(IMPORTANCE_LEVELS, key=IMPORTANCE_LEVELS.get),
            "general_novel_only": True,
        },
    }


def _build_annotation_prompt(payload: dict) -> str:
    return (
        "你是通用小说说话人标注助手。只根据输入文本和已形成的记忆判断，不使用任何外部作品知识。\n"
        "任务：给当前 chunk 内每条 dialogue 标注说话人。允许 unknown、mystery、npc_group、review，"
        "不要为了降低复核数而猜测重要角色。\n"
        "关键规则：被称呼者通常不是说话者；如果只能判断为无名低重要度群体，使用 npc_group；"
        "可能误伤重要角色时使用 review。\n"
        "只返回 JSON，格式：{\"annotations\":[{\"dialogue_id\":\"...\",\"speaker_entity_id\":\"...\","
        "\"speaker_display\":\"...\",\"speaker_status\":\"known|mystery|npc_group|unknown|review\","
        "\"confidence\":0.0,\"evidence\":[\"短证据\"],\"negative_evidence\":[],"
        "\"is_backfillable\":true,\"needs_review\":false}],\"chunk_notes\":\"...\"}\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _build_chunk_summary_prompt(payload: dict) -> str:
    return (
        "你是通用小说阅读记忆助手。任务：把刚读完的 chunk 压缩成短期摘要，供后续几个 chunk 使用。\n"
        "摘要要围绕事件、人物关系、对话轮次和仍未解决的问题；不得添加输入中没有的信息。\n"
        "只返回 JSON，格式：{\"chunk_id\":\"...\",\"summary\":\"50到450字\","
        "\"active_entities\":[\"entity_id\"],\"open_questions\":[],\"evidence_refs\":[\"paragraph_id或dialogue_id\"]}\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _build_entity_discovery_prompt(payload: dict) -> str:
    return (
        "你是通用小说实体发现助手。任务：从当前 chunk 和初标注结果中提出新增实体候选，"
        "但不要直接修改角色库。\n"
        "建 character：有明确名字、稳定身份或多次影响剧情。建 mystery：可追踪但暂未命名的个体。"
        "建 npc_group：无名群体、路人、临时职能角色、围观者或低重要度群体。\n"
        "只返回 JSON，格式：{\"new_entities\":[{\"entity_type\":\"character|mystery|npc_group\","
        "\"display_name\":\"...\",\"aliases\":[],\"importance\":\"minor|medium|major\","
        "\"summary\":\"...\",\"evidence_refs\":[],\"dialogue_count_delta\":0}],"
        "\"merge_candidates\":[{\"source_entity_id\":\"...\",\"target_entity_id\":\"...\","
        "\"reason\":\"...\",\"confidence\":0.0}]}\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _build_entity_update_prompt(payload: dict) -> str:
    return (
        "你是通用小说角色记忆维护助手。任务：更新当前 chunk 出现过的已有实体卡片。"
        "只更新输入证据支持的内容，重要性只允许 minor、medium、major；升级可以提出，降级必须谨慎。\n"
        "只返回 JSON，格式：{\"updates\":[{\"entity_id\":\"...\",\"summary\":\"...\","
        "\"importance\":\"minor|medium|major\",\"dialogue_count_delta\":0,"
        "\"latest_seen_chunk_id\":\"...\",\"relationship_updates\":[],\"evidence_refs\":[]}]}\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _build_global_summary_prompt(payload: dict) -> str:
    return (
        "你是通用小说长期摘要维护助手。任务：用旧长期摘要、当前 chunk 短期摘要和结构化事实，"
        "更新卷级长期摘要。不得凭空新增当前 chunk 没有的事实。\n"
        "只返回 JSON，格式：{\"summary\":\"50到700字\","
        "\"new_facts\":[],\"retained_facts\":[],\"dropped_facts_reason\":[]}\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _build_repair_prompt(payload: dict) -> str:
    return (
        "你是通用小说当前 chunk 回看补标助手。任务：只回看 unresolved_annotations 中的 dialogue，"
        "利用当前 chunk 摘要、更新后的长期摘要和实体记忆尝试修正。仍不能判断就保留 unknown 或 review。\n"
        "只返回 JSON，格式：{\"repairs\":[{\"dialogue_id\":\"...\","
        "\"previous_speaker_entity_id\":\"...\",\"new_speaker_entity_id\":\"...\","
        "\"speaker_display\":\"...\",\"speaker_status\":\"known|mystery|npc_group|unknown|review\","
        "\"confidence\":0.0,\"reason\":\"...\",\"stop_reason\":\"resolved|unchanged|needs_review\"}]}\n"
        "输入 JSON：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _extract_annotations(
    response: Any, *, chunk: dict, request_id: str, dry_run: bool
) -> list[dict]:
    if dry_run:
        return []
    rows = _first_list_value(response, ("annotations", "dialogues", "results", "items"))
    rows_by_id = {
        _clean_text(row.get("dialogue_id")): row for row in rows if isinstance(row, dict)
    }
    annotations: list[dict] = []
    for dialogue in chunk["dialogues"]:
        row = rows_by_id.get(dialogue["dialogue_id"])
        if row is None:
            annotations.append(_unknown_annotation(dialogue, request_id, "missing_annotation"))
        else:
            annotations.append(_normalize_annotation(row, dialogue, request_id))
    return annotations


def _normalize_annotation(row: dict, dialogue: dict, request_id: str) -> dict:
    status = _clean_status(row.get("speaker_status") or row.get("status"))
    display = _clean_text(row.get("speaker_display") or row.get("display_name"))
    entity_id = _clean_text(
        row.get("speaker_entity_id")
        or row.get("entity_id")
        or _fallback_entity_id(status, display)
    )
    confidence = _clamp(_safe_float(row.get("confidence"), 0.0), 0.0, 1.0)
    needs_review = bool(row.get("needs_review")) or status in {"unknown", "review"}
    return {
        **_dialogue_annotation_base(dialogue),
        "speaker_entity_id": entity_id,
        "speaker_display": display or _display_for_status(status),
        "speaker_status": status,
        "confidence": confidence,
        "evidence": _unique_strings(_as_list(row.get("evidence")), 6),
        "negative_evidence": _unique_strings(_as_list(row.get("negative_evidence")), 4),
        "is_backfillable": bool(row.get("is_backfillable", status in {"mystery", "unknown"})),
        "needs_review": needs_review,
        "review_reason": _clean_text(row.get("review_reason")),
        "request_id": request_id,
    }


def _unknown_annotation(dialogue: dict, request_id: str, reason: str) -> dict:
    return {
        **_dialogue_annotation_base(dialogue),
        "speaker_entity_id": "unknown",
        "speaker_display": REVIEW_LABEL,
        "speaker_status": "unknown",
        "confidence": 0.0,
        "evidence": [],
        "negative_evidence": [],
        "is_backfillable": True,
        "needs_review": True,
        "review_reason": reason,
        "request_id": request_id,
    }


def _fill_missing_annotations(*, chunk: dict, annotations: list[dict]) -> list[dict]:
    by_id = {row["dialogue_id"]: row for row in annotations}
    for dialogue in chunk["dialogues"]:
        if dialogue["dialogue_id"] not in by_id:
            by_id[dialogue["dialogue_id"]] = _unknown_annotation(
                dialogue,
                _task_request_id(chunk["chunk_id"], "annotation", "fallback"),
                "missing_after_repair",
            )
    return [by_id[dialogue["dialogue_id"]] for dialogue in chunk["dialogues"]]


def _extract_repairs(
    response: Any, *, chunk: dict, annotations: list[dict], request_id: str
) -> list[dict]:
    rows = _first_list_value(response, ("repairs", "annotations", "results", "items"))
    annotations_by_id = {row["dialogue_id"]: row for row in annotations}
    repairs: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dialogue_id = _clean_text(row.get("dialogue_id"))
        previous = annotations_by_id.get(dialogue_id)
        if previous is None:
            continue
        status = _clean_status(row.get("speaker_status") or row.get("status"))
        display = _clean_text(row.get("speaker_display") or row.get("display_name"))
        new_entity_id = _clean_text(
            row.get("new_speaker_entity_id")
            or row.get("speaker_entity_id")
            or row.get("entity_id")
            or _fallback_entity_id(status, display)
        )
        repair = {
            "chunk_id": chunk["chunk_id"],
            "dialogue_id": dialogue_id,
            "previous_speaker_entity_id": _clean_text(
                row.get("previous_speaker_entity_id")
            )
            or previous["speaker_entity_id"],
            "new_speaker_entity_id": new_entity_id,
            "speaker_display": display or _display_for_status(status),
            "speaker_status": status,
            "confidence": _clamp(_safe_float(row.get("confidence"), 0.0), 0.0, 1.0),
            "reason": _clean_text(row.get("reason")),
            "stop_reason": _clean_text(row.get("stop_reason")) or "unchanged",
            "request_id": request_id,
        }
        repairs.append(repair)
    return repairs


def _apply_repairs(annotations: list[dict], repairs: list[dict]) -> list[dict]:
    repairs_by_id = {repair["dialogue_id"]: repair for repair in repairs}
    updated: list[dict] = []
    for annotation in annotations:
        repair = repairs_by_id.get(annotation["dialogue_id"])
        if repair is None or repair["stop_reason"] not in {"resolved", "updated"}:
            updated.append(annotation)
            continue
        status = repair["speaker_status"]
        needs_review = status in {"unknown", "review"}
        updated.append(
            {
                **annotation,
                "speaker_entity_id": repair["new_speaker_entity_id"],
                "speaker_display": repair["speaker_display"],
                "speaker_status": status,
                "confidence": repair["confidence"],
                "needs_review": needs_review,
                "review_reason": "" if not needs_review else "repair_needs_review",
                "repair_trace": {
                    "request_id": repair["request_id"],
                    "reason": repair["reason"],
                    "stop_reason": repair["stop_reason"],
                },
            }
        )
    return updated


def _normalize_chunk_summary(response: Any, *, chunk: dict, dry_run: bool) -> dict:
    if dry_run or not isinstance(response, dict):
        return {}
    return {
        "chunk_id": _clean_text(response.get("chunk_id")) or chunk["chunk_id"],
        "summary": _truncate_chars(_clean_text(response.get("summary")), 700),
        "active_entities": _unique_strings(_as_list(response.get("active_entities")), 24),
        "open_questions": _unique_strings(_as_list(response.get("open_questions")), 12),
        "evidence_refs": _unique_strings(_as_list(response.get("evidence_refs")), 24),
    }


def _apply_entity_discovery(
    response: Any, *, state: ReadingV2State, chunk_id: str
) -> list[dict]:
    events: list[dict] = []
    for row in _first_list_value(response, ("new_entities", "entities", "items")):
        if not isinstance(row, dict):
            continue
        entity_type = _clean_entity_type(row.get("entity_type"))
        display_name = _clean_text(row.get("display_name") or row.get("speaker_display"))
        if not entity_type or not display_name:
            continue
        table = _entity_table(state, entity_type)
        entity_id = _clean_text(row.get("entity_id"))
        if not entity_id:
            entity_id = _find_entity_id_by_display(table, display_name)
        if not entity_id:
            entity_id = _next_entity_id(state, entity_type)
        old = table.get(entity_id, {})
        card = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "display_name": old.get("display_name") or display_name,
            "aliases": _unique_strings(
                [*old.get("aliases", []), *_as_list(row.get("aliases"))], 12
            ),
            "importance": _merge_importance(
                old.get("importance"), _clean_importance(row.get("importance"))
            ),
            "summary": _truncate_chars(
                _clean_text(row.get("summary")) or old.get("summary", ""),
                180 if entity_type != "npc_group" else 120,
            ),
            "evidence_refs": _unique_strings(
                [*old.get("evidence_refs", []), *_as_list(row.get("evidence_refs"))],
                24,
            ),
            "dialogue_count": int(old.get("dialogue_count", 0))
            + max(0, int(_safe_float(row.get("dialogue_count_delta"), 0))),
            "first_seen_chunk_id": old.get("first_seen_chunk_id") or chunk_id,
            "latest_seen_chunk_id": chunk_id,
        }
        table[entity_id] = card
        events.append(
            {
                "chunk_id": chunk_id,
                "event_type": "entity_discovery",
                "entity_id": entity_id,
                "entity_type": entity_type,
                "display_name": card["display_name"],
            }
        )
    return events


def _apply_entity_updates(
    response: Any, *, state: ReadingV2State, chunk_id: str
) -> list[dict]:
    events: list[dict] = []
    for row in _first_list_value(response, ("updates", "entities", "items")):
        if not isinstance(row, dict):
            continue
        entity_id = _clean_text(row.get("entity_id"))
        if not entity_id:
            continue
        table = _find_entity_table_by_id(state, entity_id)
        if table is None:
            continue
        card = table[entity_id]
        summary = _clean_text(row.get("summary"))
        if summary:
            card["summary"] = _truncate_chars(
                summary, 180 if card["entity_type"] != "npc_group" else 120
            )
        card["importance"] = _merge_importance(
            card.get("importance"), _clean_importance(row.get("importance"))
        )
        card["dialogue_count"] = int(card.get("dialogue_count", 0)) + max(
            0, int(_safe_float(row.get("dialogue_count_delta"), 0))
        )
        card["latest_seen_chunk_id"] = (
            _clean_text(row.get("latest_seen_chunk_id")) or chunk_id
        )
        card["relationship_updates"] = _unique_strings(
            [
                *card.get("relationship_updates", []),
                *_as_list(row.get("relationship_updates")),
            ],
            20,
        )
        card["evidence_refs"] = _unique_strings(
            [*card.get("evidence_refs", []), *_as_list(row.get("evidence_refs"))],
            24,
        )
        events.append(
            {
                "chunk_id": chunk_id,
                "event_type": "entity_update",
                "entity_id": entity_id,
                "entity_type": card.get("entity_type"),
            }
        )
    return events


def _apply_global_summary(
    response: Any, *, state: ReadingV2State, chunk_id: str
) -> None:
    if not isinstance(response, dict):
        return
    summary = _clean_text(response.get("summary") or response.get("global_summary"))
    if summary:
        state.global_summary = _truncate_chars(summary, 900)
    for key in ("new_facts", "retained_facts"):
        for fact in _as_list(response.get(key)):
            cleaned = _clean_text(fact)
            if cleaned:
                state.facts.append(
                    {"chunk_id": chunk_id, "fact_type": key, "text": cleaned}
                )


def _ensure_entities_from_annotations(
    annotations: list[dict], state: ReadingV2State, chunk_id: str
) -> None:
    for annotation in annotations:
        status = annotation["speaker_status"]
        if status not in {"known", "mystery", "npc_group"}:
            continue
        entity_type = "character" if status == "known" else status
        table = _entity_table(state, entity_type)
        entity_id = annotation["speaker_entity_id"]
        if entity_id in table:
            continue
        display = _clean_text(annotation.get("speaker_display"))
        if not display or display == REVIEW_LABEL:
            continue
        table[entity_id] = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "display_name": display,
            "aliases": [],
            "importance": "medium" if entity_type == "character" else "minor",
            "summary": "",
            "evidence_refs": _unique_strings(annotation.get("evidence", []), 6),
            "dialogue_count": 1,
            "first_seen_chunk_id": chunk_id,
            "latest_seen_chunk_id": chunk_id,
        }


def _append_entity_events(
    *, state: ReadingV2State, chunk_id: str, events: list[dict]
) -> None:
    for event in events:
        state.entity_events.append({"chunk_id": chunk_id, **event})


def _write_prompt_length_reports(*, reading_dir: Path, reports: list[dict]) -> dict:
    write_jsonl(reading_dir / "prompt_length_report.jsonl", reports)
    summary = _prompt_summary(reports)
    write_json(reading_dir / "prompt_length_report.json", {**summary, "reports": reports})
    return summary


def _prompt_summary(reports: list[dict]) -> dict:
    estimated = [int(row["estimated_tokens"]) for row in reports]
    return {
        "prompt_count": len(reports),
        "max_estimated_tokens": max(estimated) if estimated else 0,
        "avg_estimated_tokens": round(sum(estimated) / len(estimated), 2)
        if estimated
        else 0,
        "near_limit_count": sum(1 for row in reports if row["status"] == "near_limit"),
        "over_limit_count": sum(1 for row in reports if row["status"] == "over_limit"),
        "task_counts": {
            task: sum(1 for row in reports if row["task"] == task)
            for task in READING_V2_TASKS
        },
    }


def _write_memory_outputs(
    *, memory_dir: Path, state: ReadingV2State, volume_meta: dict
) -> None:
    write_json(
        memory_dir / "global_summary.json",
        {
            "volume_id": volume_meta.get("volume_id"),
            "summary": state.global_summary,
            "chunk_summary_count": len(state.chunk_summaries),
        },
    )
    write_jsonl(memory_dir / "chunk_summaries.jsonl", state.chunk_summaries)
    write_jsonl(memory_dir / "facts.jsonl", state.facts)
    write_jsonl(memory_dir / "characters.jsonl", state.characters.values())
    write_jsonl(memory_dir / "mysteries.jsonl", state.mysteries.values())
    write_jsonl(memory_dir / "npc_groups.jsonl", state.npc_groups.values())
    write_jsonl(memory_dir / "entity_events.jsonl", state.entity_events)


def _write_annotation_outputs(
    *,
    annotation_dir: Path,
    paragraphs: list[dict],
    annotations: list[dict],
    repairs: list[dict],
    dry_run: bool,
) -> None:
    write_jsonl(annotation_dir / "annotations.jsonl", annotations)
    write_jsonl(annotation_dir / "repairs.jsonl", repairs)
    write_jsonl(
        annotation_dir / "review_queue.jsonl",
        [row for row in annotations if row.get("needs_review")],
    )
    if not dry_run:
        _write_labeled_text(
            paragraphs=paragraphs,
            annotations=annotations,
            output_path=annotation_dir / "final_labeled.txt",
        )


def _write_labeled_text(
    *, paragraphs: list[dict], annotations: list[dict], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_labeled_text(paragraphs, annotations), encoding="utf-8", newline="\n"
    )


def _render_labeled_text(paragraphs: list[dict], annotations: list[dict]) -> str:
    by_paragraph: dict[str, list[dict]] = {}
    for annotation in annotations:
        by_paragraph.setdefault(annotation["paragraph_id"], []).append(annotation)

    rendered: list[str] = []
    for paragraph in sorted(paragraphs, key=lambda row: int(row.get("paragraph_index", 0))):
        text = str(paragraph.get("text", ""))
        paragraph_annotations = sorted(
            by_paragraph.get(paragraph["paragraph_id"], []),
            key=lambda row: int(row.get("char_start", 0)),
            reverse=True,
        )
        for annotation in paragraph_annotations:
            label = _label_for_annotation(annotation)
            if not label:
                continue
            insert_at = max(0, min(int(annotation.get("char_start", 0)), len(text)))
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
    if status in {"known", "mystery", "npc_group"} and display:
        return display
    return REVIEW_LABEL


def _prompt_length_report(
    *,
    chunk: dict,
    task: str,
    prompt: str,
    prompt_path: Path,
    output_dir: Path,
    config: ReadingV2Config,
) -> dict:
    estimated = estimate_prompt_tokens(prompt)
    component_counts = _prompt_component_counts(prompt)
    return {
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk["chunk_index"],
        "task": task,
        "prompt_path": _relative_output_path(prompt_path, output_dir),
        "char_count": len(prompt),
        "estimated_tokens": estimated,
        "target_token_limit": config.max_prompt_tokens,
        "hard_token_limit": config.hard_prompt_tokens,
        "status": prompt_length_status(
            estimated, config.max_prompt_tokens, config.hard_prompt_tokens
        ),
        **component_counts,
    }


def _prompt_component_counts(prompt: str) -> dict:
    chinese_chars = 0
    ascii_chars = 0
    json_punctuation = 0
    other_chars = 0
    for char in prompt:
        if "\u4e00" <= char <= "\u9fff":
            chinese_chars += 1
        elif char.isascii() and char.isalnum():
            ascii_chars += 1
        elif char in "{}[],:\"'，。！？；：、":
            json_punctuation += 1
        else:
            other_chars += 1
    return {
        "chinese_chars": chinese_chars,
        "ascii_chars": ascii_chars,
        "json_punctuation": json_punctuation,
        "other_chars": other_chars,
    }


def _memory_snapshot_for_output(state: ReadingV2State) -> dict:
    return {
        "characters": list(state.characters.values()),
        "mysteries": list(state.mysteries.values()),
        "npc_groups": list(state.npc_groups.values()),
    }


def _candidate_entity_cards(state: ReadingV2State, config: ReadingV2Config) -> list[dict]:
    rows = [
        *_entity_prompt_cards(state.characters.values()),
        *_entity_prompt_cards(state.mysteries.values()),
        *_entity_prompt_cards(state.npc_groups.values()),
    ]
    rows.sort(
        key=lambda row: (
            IMPORTANCE_LEVELS.get(row.get("importance", "minor"), 1),
            _chunk_number(row.get("latest_seen_chunk_id")),
        ),
        reverse=True,
    )
    return rows[: max(0, config.max_candidate_entities)]


def _entity_prompt_cards(rows: Any) -> list[dict]:
    cards: list[dict] = []
    for row in rows:
        cards.append(
            {
                "entity_id": row.get("entity_id"),
                "entity_type": row.get("entity_type"),
                "display_name": row.get("display_name"),
                "aliases": row.get("aliases", [])[:6],
                "importance": row.get("importance", "minor"),
                "summary": row.get("summary", ""),
                "latest_seen_chunk_id": row.get("latest_seen_chunk_id", ""),
                "dialogue_count": row.get("dialogue_count", 0),
            }
        )
    return cards


def _paragraph_card(row: dict) -> dict:
    return {
        "paragraph_id": row.get("paragraph_id"),
        "paragraph_index": row.get("paragraph_index"),
        "chapter_id": row.get("chapter_id", ""),
        "chapter_index": row.get("chapter_index", 0),
        "chapter_title": row.get("chapter_title", ""),
        "scene_id": row.get("scene_id", ""),
        "scene_index": row.get("scene_index", 0),
        "text": row.get("text", ""),
    }


def _dialogue_card(row: dict) -> dict:
    return {
        "dialogue_id": row.get("dialogue_id"),
        "dialogue_index": row.get("dialogue_index"),
        "paragraph_id": row.get("paragraph_id"),
        "paragraph_index": row.get("paragraph_index"),
        "local_dialogue_index": row.get("local_dialogue_index", 0),
        "dialogue_kind": _dialogue_kind(row),
        "text": row.get("text", ""),
        "quote_text": row.get("quote_text", ""),
        "char_start": row.get("char_start", 0),
        "char_end": row.get("char_end", 0),
    }


def _dialogue_annotation_base(dialogue: dict) -> dict:
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "dialogue_index": dialogue.get("dialogue_index", 0),
        "volume_id": dialogue.get("volume_id", ""),
        "chapter_id": dialogue.get("chapter_id", ""),
        "chapter_index": dialogue.get("chapter_index", 0),
        "chapter_title": dialogue.get("chapter_title", ""),
        "scene_id": dialogue.get("scene_id", ""),
        "scene_index": dialogue.get("scene_index", 0),
        "paragraph_id": dialogue["paragraph_id"],
        "paragraph_index": dialogue.get("paragraph_index", 0),
        "local_dialogue_index": dialogue.get("local_dialogue_index", 0),
        "text": dialogue.get("text", ""),
        "quote_text": dialogue.get("quote_text", ""),
        "dialogue_kind": _dialogue_kind(dialogue),
        "char_start": dialogue.get("char_start", 0),
        "char_end": dialogue.get("char_end", 0),
    }


def _task_request_id(chunk_id: str, task: str, model: str) -> str:
    return f"{chunk_id}--{task}--{_safe_model_name(model)}"


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


def _first_list_value(row: Any, keys: tuple[str, ...]) -> list:
    if isinstance(row, list):
        return row
    if isinstance(row, dict):
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                return value
    return []


def _entity_table(state: ReadingV2State, entity_type: str) -> dict[str, dict]:
    if entity_type == "character":
        return state.characters
    if entity_type == "mystery":
        return state.mysteries
    return state.npc_groups


def _find_entity_table_by_id(
    state: ReadingV2State, entity_id: str
) -> dict[str, dict] | None:
    for table in (state.characters, state.mysteries, state.npc_groups):
        if entity_id in table:
            return table
    return None


def _next_entity_id(state: ReadingV2State, entity_type: str) -> str:
    if entity_type == "character":
        value = f"char_v2_{state.next_character_id:06d}"
        state.next_character_id += 1
        return value
    if entity_type == "mystery":
        value = f"mystery_v2_{state.next_mystery_id:06d}"
        state.next_mystery_id += 1
        return value
    value = f"npc_group_v2_{state.next_npc_group_id:06d}"
    state.next_npc_group_id += 1
    return value


def _find_entity_id_by_display(table: dict[str, dict], display_name: str) -> str:
    for entity_id, card in table.items():
        names = [card.get("display_name"), *card.get("aliases", [])]
        if display_name in {_clean_text(name) for name in names}:
            return entity_id
    return ""


def _fallback_entity_id(status: str, display: str) -> str:
    if status == "unknown":
        return "unknown"
    if status == "review":
        return "review"
    if not display:
        return status
    return f"{status}:{_clean_identifier(display)}"


def _display_for_status(status: str) -> str:
    if status == "npc_group":
        return "无名群体"
    if status == "unknown":
        return "未知"
    return REVIEW_LABEL


def _merge_importance(old_value: Any, new_value: str) -> str:
    old = _clean_importance(old_value)
    new = _clean_importance(new_value)
    if IMPORTANCE_LEVELS[new] >= IMPORTANCE_LEVELS[old]:
        return new
    return old


def _clean_status(value: Any) -> str:
    status = _clean_text(value).lower()
    if status == "npc":
        status = "npc_group"
    if status in SPEAKER_STATUSES:
        return status
    return "review"


def _clean_entity_type(value: Any) -> str:
    entity_type = _clean_text(value).lower()
    if entity_type == "known":
        entity_type = "character"
    if entity_type == "npc":
        entity_type = "npc_group"
    if entity_type in {"character", "mystery", "npc_group"}:
        return entity_type
    return ""


def _clean_importance(value: Any) -> str:
    importance = _clean_text(value).lower()
    if importance in IMPORTANCE_LEVELS:
        return importance
    return "minor"


def _dialogue_kind(dialogue: dict) -> str:
    kind = _clean_text(dialogue.get("dialogue_kind"))
    return kind or "standalone"


def _first_text(rows: list[dict], key: str) -> str:
    for row in rows:
        value = _clean_text(row.get(key))
        if value:
            return value
    return ""


def _first_int(rows: list[dict], key: str) -> int:
    for row in rows:
        try:
            return int(row.get(key))
        except (TypeError, ValueError):
            continue
    return 0


def _read_required_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL not found: {path}")
    return list(read_jsonl(path))


def _read_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_strings(values: Any, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _as_list(values):
        cleaned = _clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
            if limit is not None and len(result) >= limit:
                break
    return result


def _truncate_chars(value: str, max_length: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 1)] + "…"


def _chunk_number(chunk_id: Any) -> int:
    match = re.search(r"-r(\d+)$", _clean_text(chunk_id))
    return int(match.group(1)) if match else 0


def _relative_output_path(path: Path, output_dir: Path) -> str:
    try:
        return _output_path_for_log(path.relative_to(output_dir))
    except ValueError:
        return _output_path_for_log(path)


def _output_path_for_log(path: Path) -> str:
    return str(path).replace("\\", "/")
