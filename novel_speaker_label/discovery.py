from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jsonl import read_json, read_jsonl, write_json, write_jsonl
from .ollama_client import OllamaClient, OllamaConfig

GENERIC_MERGE_KEYS = {
    "中间人",
    "交易者",
    "同行",
    "商人",
    "小毛头",
    "市井无赖",
    "年轻人",
    "年轻商人",
    "开店者",
    "情报贩子",
    "提议者",
    "新手",
    "旅行商人",
    "旅行商人亲戚",
    "经验丰富的商人",
    "行商人",
    "老板",
    "小的",
    "男子",
    "女人",
    "店员",
    "女店员",
    "店老板",
    "骑士",
    "农夫",
    "马夫",
    "采购人员",
    "商行代表",
    "行长",
    "丰收之神",
    "贤狼",
    "神秘少女",
    "咱",
    "汝",
}


@dataclass(frozen=True)
class DiscoveryConfig:
    output_dir: Path
    model: str = "qwen3:32b"
    ollama_host: str = "http://127.0.0.1:11434"
    timeout: int = 1800
    temperature: float = 0.0
    num_predict: int = 4096
    dry_run: bool = False
    overwrite_cache: bool = False
    max_known_characters: int = 30
    max_paragraphs_per_request: int = 30
    max_dialogues_per_request: int = 40
    continue_on_error: bool = True


class CharacterStore:
    def __init__(self) -> None:
        self.characters: list[dict] = []
        self.key_to_id: dict[str, str] = {}

    def add_or_update(self, candidate: dict, scene_id: str) -> dict | None:
        display_name = _clean_name(candidate.get("display_name") or candidate.get("name"))
        if not display_name:
            return None

        raw_aliases = {
            _clean_name(alias)
            for alias in _as_list(candidate.get("aliases"))
            if _clean_name(alias)
        }
        aliases = {alias for alias in raw_aliases if _is_safe_merge_key(alias)}
        titles = {
            _clean_name(title)
            for title in _as_list(candidate.get("titles"))
            if _clean_name(title)
        }
        titles.update(alias for alias in raw_aliases if alias not in aliases)
        merge_keys = _merge_keys(display_name, aliases)
        existing_id = self._find_existing_id(merge_keys)

        if existing_id is None:
            entity = {
                "entity_id": f"char_{len(self.characters) + 1:04d}",
                "display_name": display_name,
                "aliases": sorted(aliases)[:20],
                "titles": sorted(titles)[:20],
                "description": _clean_text(candidate.get("description")),
                "speech_style": _clean_text(candidate.get("speech_style")),
                "relationship_hints": _unique_strings(
                    _as_list(candidate.get("relationship_hints")), limit=80
                ),
                "evidence": [],
                "first_seen_scene_id": scene_id,
                "latest_seen_scene_id": scene_id,
                "confidence": _safe_float(candidate.get("confidence"), default=0.5),
            }
            self.characters.append(entity)
        else:
            entity = self._by_id(existing_id)
            entity["aliases"] = sorted(set(entity["aliases"]) | aliases)[:20]
            entity["titles"] = sorted(set(entity["titles"]) | titles)[:20]
            entity["relationship_hints"] = _unique_strings(
                entity["relationship_hints"]
                + _as_list(candidate.get("relationship_hints")),
                limit=80,
            )
            entity["description"] = _join_field(
                entity["description"], _clean_text(candidate.get("description"))
            )
            entity["speech_style"] = _join_field(
                entity["speech_style"], _clean_text(candidate.get("speech_style"))
            )
            entity["latest_seen_scene_id"] = scene_id
            entity["confidence"] = max(
                entity["confidence"], _safe_float(candidate.get("confidence"), default=0.5)
            )

        for key in _merge_keys(display_name, set(entity["aliases"])):
            self.key_to_id[key] = entity["entity_id"]

        for evidence in _as_list(candidate.get("evidence")):
            cleaned = _clean_text(evidence)
            if cleaned:
                entity["evidence"].append({"scene_id": scene_id, "text": cleaned})
        entity["evidence"] = entity["evidence"][-8:]
        return entity

    def _by_id(self, entity_id: str) -> dict:
        for entity in self.characters:
            if entity["entity_id"] == entity_id:
                return entity
        raise KeyError(entity_id)

    def _find_existing_id(self, keys: set[str]) -> str | None:
        for key in keys:
            if key in self.key_to_id:
                return self.key_to_id[key]
        for key in keys:
            for existing_key, entity_id in self.key_to_id.items():
                if _names_look_related(key, existing_key):
                    return entity_id
        return None

    def snapshot_for_prompt(self, limit: int) -> list[dict]:
        rows = self.characters[-limit:]
        return [
            {
                "entity_id": row["entity_id"],
                "display_name": row["display_name"],
                "aliases": row["aliases"],
                "titles": row["titles"],
                "description": row["description"],
            }
            for row in rows
        ]


def discover_volume(config: DiscoveryConfig) -> dict:
    volume_meta = read_json(config.output_dir / "volume.json")
    preprocess_dir = config.output_dir / "preprocess"
    paragraphs = list(read_jsonl(preprocess_dir / "paragraphs.jsonl"))
    dialogues = list(read_jsonl(preprocess_dir / "dialogues.jsonl"))
    scenes = list(read_jsonl(preprocess_dir / "scenes.jsonl"))

    paragraphs_by_id = {row["paragraph_id"]: row for row in paragraphs}
    dialogues_by_scene = _group_by(dialogues, "scene_id")
    discovery_dir = config.output_dir / "discovery"
    cache_dir = discovery_dir / "cache"
    failure_dir = discovery_dir / "failures"
    prompt_dir = discovery_dir / "prompts"
    raw_dir = discovery_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(
        OllamaConfig(
            host=config.ollama_host,
            model=config.model,
            timeout=config.timeout,
            temperature=config.temperature,
            num_predict=config.num_predict,
        )
    )
    character_store = CharacterStore()
    scene_discoveries: list[dict] = []
    raw_responses: list[dict] = []
    mystery_entities: list[dict] = []
    alias_rows: list[dict] = []
    failed_requests: list[dict] = []
    prompt_count = 0
    request_jobs = _build_request_jobs(
        scenes=scenes,
        paragraphs_by_id=paragraphs_by_id,
        dialogues_by_scene=dialogues_by_scene,
        max_paragraphs=config.max_paragraphs_per_request,
        max_dialogues=config.max_dialogues_per_request,
    )

    for request_number, job in enumerate(request_jobs, start=1):
        print(
            "[discover] "
            f"{request_number}/{len(request_jobs)} {job['request_id']} "
            f"paragraphs={len(job['paragraphs'])} dialogues={len(job['dialogues'])}",
            flush=True,
        )

        prompt_payload = _build_scene_payload(
            job=job,
            known_characters=character_store.snapshot_for_prompt(
                config.max_known_characters
            ),
            volume_meta=volume_meta,
        )
        prompt = _build_discovery_prompt(prompt_payload)
        prompt_path = prompt_dir / f"{job['request_id']}.txt"
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
        prompt_count += 1

        cached_path = cache_dir / f"{job['request_id']}.json"
        if config.dry_run:
            continue
        if cached_path.exists() and not config.overwrite_cache:
            discovery = read_json(cached_path)
            response_text = json.dumps(discovery, ensure_ascii=False)
        else:
            try:
                response_text = client.generate(prompt)
                (raw_dir / f"{job['request_id']}.txt").write_text(
                    response_text, encoding="utf-8", newline="\n"
                )
                discovery = _parse_json_response(response_text)
                write_json(cached_path, discovery)
            except Exception as exc:
                failure = {
                    "request_id": job["request_id"],
                    "scene_id": job["scene_id"],
                    "chunk_index": job["chunk_index"],
                    "model": config.model,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "prompt_path": str(prompt_path),
                    "paragraph_count": len(job["paragraphs"]),
                    "dialogue_count": len(job["dialogues"]),
                }
                write_json(failure_dir / f"{job['request_id']}.json", failure)
                failed_requests.append(failure)
                print(
                    "[discover] failed "
                    f"{job['request_id']}: {failure['error_type']}: {failure['error']}",
                    flush=True,
                )
                if not config.continue_on_error:
                    raise RuntimeError(
                        f"Discovery request failed: {job['request_id']}. "
                        f"Prompt is saved at {prompt_path}"
                    ) from exc
                continue

        model_scene_id = discovery.get("scene_id")
        discovery["scene_id"] = job["request_id"]
        discovery["parent_scene_id"] = job["scene_id"]
        discovery["model_scene_id"] = model_scene_id
        discovery["chunk_index"] = job["chunk_index"]
        discovery["chunk_count"] = job["chunk_count"]
        scene_discoveries.append(discovery)
        raw_responses.append(
            {
                "request_id": job["request_id"],
                "scene_id": job["scene_id"],
                "chunk_index": job["chunk_index"],
                "model": config.model,
                "response": response_text,
            }
        )

        for candidate in _as_list(discovery.get("character_candidates")):
            entity = character_store.add_or_update(candidate, job["request_id"])
            if entity:
                for alias in entity["aliases"]:
                    alias_rows.append(
                        {
                            "entity_id": entity["entity_id"],
                            "display_name": entity["display_name"],
                            "alias": alias,
                            "source_scene_id": job["request_id"],
                        }
                    )
        for mystery in _as_list(discovery.get("mystery_entities")):
            mystery_entities.append(_normalize_mystery_entity(mystery, job["request_id"]))

    if config.dry_run:
        write_json(
            discovery_dir / "dry_run.json",
            {
                "volume_id": volume_meta["volume_id"],
                "model": config.model,
                "prompt_count": prompt_count,
                "request_count": len(request_jobs),
                "message": "Prompts were generated; no Ollama requests were made.",
            },
        )
    else:
        write_jsonl(discovery_dir / "scene_discoveries.jsonl", scene_discoveries)
        write_jsonl(discovery_dir / "raw_responses.jsonl", raw_responses)
        write_jsonl(discovery_dir / "failed_requests.jsonl", failed_requests)
        _write_memory_outputs(
            output_dir=config.output_dir,
            characters=character_store.characters,
            scenes=scene_discoveries,
            mystery_entities=mystery_entities,
            alias_rows=alias_rows,
        )

    run_summary = {
        "volume_id": volume_meta["volume_id"],
        "model": config.model,
        "dry_run": config.dry_run,
        "processed_scene_count": prompt_count if config.dry_run else len(scene_discoveries),
        "prompt_count": prompt_count,
        "request_count": len(request_jobs),
        "failed_request_count": len(failed_requests),
        "character_count": len(character_store.characters),
        "mystery_entity_count": len(mystery_entities),
    }
    write_json(discovery_dir / "run_summary.json", run_summary)
    return run_summary


def _build_scene_payload(
    job: dict,
    known_characters: list[dict],
    volume_meta: dict,
) -> dict:
    return {
        "volume_id": volume_meta["volume_id"],
        "scene": {
            "request_id": job["request_id"],
            "scene_id": job["scene_id"],
            "chapter_id": job["chapter_id"],
            "chapter_title": job["chapter_title"],
            "chunk_index": job["chunk_index"],
            "chunk_count": job["chunk_count"],
        },
        "paragraphs": [
            {
                "paragraph_id": row["paragraph_id"],
                "text": row["text"],
            }
            for row in job["paragraphs"]
        ],
        "dialogues": [
            {
                "dialogue_id": row["dialogue_id"],
                "paragraph_id": row["paragraph_id"],
                "text": row["text"],
            }
            for row in job["dialogues"]
        ],
        "known_characters": known_characters,
    }


def _build_request_jobs(
    scenes: list[dict],
    paragraphs_by_id: dict[str, dict],
    dialogues_by_scene: dict[str, list[dict]],
    max_paragraphs: int,
    max_dialogues: int,
) -> list[dict]:
    jobs: list[dict] = []
    for scene in scenes:
        scene_dialogues = dialogues_by_scene.get(scene["scene_id"], [])
        if not scene_dialogues:
            continue

        scene_jobs = _split_scene_into_requests(
            scene=scene,
            paragraphs_by_id=paragraphs_by_id,
            dialogues=scene_dialogues,
            max_paragraphs=max_paragraphs,
            max_dialogues=max_dialogues,
        )
        for index, job in enumerate(scene_jobs, start=1):
            job["chunk_index"] = index
            job["chunk_count"] = len(scene_jobs)
            job["request_id"] = f"{scene['scene_id']}-r{index:03d}"
            jobs.append(job)
    return jobs


def _split_scene_into_requests(
    scene: dict,
    paragraphs_by_id: dict[str, dict],
    dialogues: list[dict],
    max_paragraphs: int,
    max_dialogues: int,
) -> list[dict]:
    paragraphs = [
        paragraphs_by_id[paragraph_id]
        for paragraph_id in scene["paragraph_ids"]
        if paragraph_id in paragraphs_by_id
    ]
    dialogues_by_paragraph = _group_by(dialogues, "paragraph_id")
    max_paragraphs = max(1, max_paragraphs)
    max_dialogues = max(1, max_dialogues)

    jobs: list[dict] = []
    current_paragraphs: list[dict] = []
    current_dialogues: list[dict] = []

    def flush_current() -> None:
        if not current_dialogues:
            current_paragraphs.clear()
            return
        jobs.append(
            {
                "scene_id": scene["scene_id"],
                "chapter_id": scene["chapter_id"],
                "chapter_title": scene["chapter_title"],
                "paragraphs": list(current_paragraphs),
                "dialogues": list(current_dialogues),
            }
        )
        current_paragraphs.clear()
        current_dialogues.clear()

    for paragraph in paragraphs:
        paragraph_dialogues = dialogues_by_paragraph.get(paragraph["paragraph_id"], [])
        would_exceed_paragraphs = len(current_paragraphs) >= max_paragraphs
        would_exceed_dialogues = (
            current_dialogues
            and len(current_dialogues) + len(paragraph_dialogues) > max_dialogues
        )
        if current_paragraphs and (would_exceed_paragraphs or would_exceed_dialogues):
            flush_current()

        current_paragraphs.append(paragraph)
        current_dialogues.extend(paragraph_dialogues)

    flush_current()
    return jobs


def _build_discovery_prompt(payload: dict) -> str:
    return (
        "你是轻小说说话人标注项目的阶段 1：角色发现与建库助手。\n"
        "你的任务不是给每句台词最终打标签，而是从当前场景中发现角色、别名、称谓、"
        "关系线索、说话风格线索、场景摘要，以及尚未命名但可追踪的神秘人物。\n\n"
        "规则：\n"
        "1. 只输出严格 JSON，不要 Markdown，不要解释性前后缀。\n"
        "2. scene_id 必须填写输入 scene.request_id。\n"
        "3. 不要编造没有文本证据的人名。\n"
        "4. 如果人物未命名但看起来可在后文追踪，放入 mystery_entities。\n"
        "5. character_candidates 只放可以从文本中看到名字、称谓或稳定身份线索的人物。\n"
        "6. titles 只放身份/职业/称谓，不要把职业称谓当成人物别名。\n"
        "7. evidence 使用 paragraph_id 或 dialogue_id 相关的短证据，不要长篇复制原文。\n\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "scene_id": "string",\n'
        '  "scene_summary": "string",\n'
        '  "active_characters": ["string"],\n'
        '  "character_candidates": [\n'
        "    {\n"
        '      "display_name": "string",\n'
        '      "aliases": ["string"],\n'
        '      "titles": ["string"],\n'
        '      "description": "string",\n'
        '      "speech_style": "string",\n'
        '      "relationship_hints": ["string"],\n'
        '      "evidence": ["string"],\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "mystery_entities": [\n'
        "    {\n"
        '      "temporary_name": "string",\n'
        '      "description": "string",\n'
        '      "evidence": ["string"],\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "relationships": ["string"],\n'
        '  "notes": "string"\n'
        "}\n\n"
        "输入场景 JSON：\n"
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


def _write_memory_outputs(
    output_dir: Path,
    characters: list[dict],
    scenes: list[dict],
    mystery_entities: list[dict],
    alias_rows: list[dict],
) -> None:
    memory_dir = output_dir / "memory"
    write_jsonl(memory_dir / "semantic" / "characters.jsonl", characters)
    write_jsonl(memory_dir / "episodic" / "scenes.jsonl", scenes)
    write_jsonl(memory_dir / "mystery_entities.jsonl", mystery_entities)
    write_jsonl(memory_dir / "aliases.jsonl", _dedupe_alias_rows(alias_rows))


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("：:，,。 .「」『』【】[]()（）")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _merge_keys(display_name: str, aliases: set[str]) -> set[str]:
    keys = {display_name}
    keys.update(alias for alias in aliases if _is_safe_merge_key(alias))
    return {key for key in keys if key}


def _is_safe_merge_key(value: str) -> bool:
    if not value:
        return False
    if value in GENERIC_MERGE_KEYS:
        return False
    if len(value) <= 1:
        return False
    return True


def _names_look_related(left: str, right: str) -> bool:
    if not (_is_safe_merge_key(left) and _is_safe_merge_key(right)):
        return False
    if left == right:
        return True
    if len(left) < 3 or len(right) < 3:
        return False
    return left.endswith(right) or right.endswith(left)


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


def _join_field(existing: str, new_value: str, max_parts: int = 12) -> str:
    if not new_value:
        return existing
    if not existing:
        return new_value
    parts = [part.strip() for part in existing.split(" / ") if part.strip()]
    if new_value not in parts:
        parts.append(new_value)
    return " / ".join(parts[-max_parts:])


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_mystery_entity(row: dict, scene_id: str) -> dict:
    return {
        "temporary_name": _clean_name(row.get("temporary_name")) or "mystery_unknown",
        "description": _clean_text(row.get("description")),
        "evidence": _unique_strings(_as_list(row.get("evidence"))),
        "source_scene_id": scene_id,
        "confidence": _safe_float(row.get("confidence"), default=0.5),
    }


def _dedupe_alias_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for row in rows:
        key = (row["entity_id"], row["alias"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result
