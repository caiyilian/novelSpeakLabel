from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jsonl import read_json, read_jsonl, write_json, write_jsonl
from .ollama_client import OllamaClient, OllamaConfig


@dataclass(frozen=True)
class DiscoveryConfig:
    output_dir: Path
    model: str = "qwen3:32b"
    ollama_host: str = "http://127.0.0.1:11434"
    timeout: int = 120
    temperature: float = 0.0
    num_predict: int = 8192
    dry_run: bool = False
    overwrite_cache: bool = False
    max_known_characters: int = 30


class CharacterStore:
    def __init__(self) -> None:
        self.characters: list[dict] = []
        self.key_to_id: dict[str, str] = {}

    def add_or_update(self, candidate: dict, scene_id: str) -> dict | None:
        display_name = _clean_name(candidate.get("display_name") or candidate.get("name"))
        if not display_name:
            return None

        aliases = {
            _clean_name(alias)
            for alias in _as_list(candidate.get("aliases"))
            if _clean_name(alias)
        }
        titles = {
            _clean_name(title)
            for title in _as_list(candidate.get("titles"))
            if _clean_name(title)
        }
        keys = {display_name, *aliases, *titles}
        existing_id = next((self.key_to_id[key] for key in keys if key in self.key_to_id), None)

        if existing_id is None:
            entity = {
                "entity_id": f"char_{len(self.characters) + 1:04d}",
                "display_name": display_name,
                "aliases": sorted(aliases),
                "titles": sorted(titles),
                "description": _clean_text(candidate.get("description")),
                "speech_style": _clean_text(candidate.get("speech_style")),
                "relationship_hints": _unique_strings(
                    _as_list(candidate.get("relationship_hints"))
                ),
                "evidence": [],
                "first_seen_scene_id": scene_id,
                "latest_seen_scene_id": scene_id,
                "confidence": _safe_float(candidate.get("confidence"), default=0.5),
            }
            self.characters.append(entity)
        else:
            entity = self._by_id(existing_id)
            entity["aliases"] = sorted(set(entity["aliases"]) | aliases)
            entity["titles"] = sorted(set(entity["titles"]) | titles)
            entity["relationship_hints"] = _unique_strings(
                entity["relationship_hints"]
                + _as_list(candidate.get("relationship_hints"))
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

        for key in {display_name, *entity["aliases"], *entity["titles"]}:
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
    prompt_dir = discovery_dir / "prompts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

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
    prompt_count = 0

    for scene in scenes:
        scene_dialogues = dialogues_by_scene.get(scene["scene_id"], [])
        if not scene_dialogues:
            continue

        prompt_payload = _build_scene_payload(
            scene=scene,
            paragraphs_by_id=paragraphs_by_id,
            dialogues=scene_dialogues,
            known_characters=character_store.snapshot_for_prompt(
                config.max_known_characters
            ),
            volume_meta=volume_meta,
        )
        prompt = _build_discovery_prompt(prompt_payload)
        prompt_path = prompt_dir / f"{scene['scene_id']}.txt"
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
        prompt_count += 1

        cached_path = cache_dir / f"{scene['scene_id']}.json"
        if config.dry_run:
            continue
        if cached_path.exists() and not config.overwrite_cache:
            discovery = read_json(cached_path)
            response_text = json.dumps(discovery, ensure_ascii=False)
        else:
            response_text = client.generate(prompt)
            discovery = _parse_json_response(response_text)
            write_json(cached_path, discovery)

        discovery.setdefault("scene_id", scene["scene_id"])
        scene_discoveries.append(discovery)
        raw_responses.append(
            {
                "scene_id": scene["scene_id"],
                "model": config.model,
                "response": response_text,
            }
        )

        for candidate in _as_list(discovery.get("character_candidates")):
            entity = character_store.add_or_update(candidate, scene["scene_id"])
            if entity:
                for alias in entity["aliases"]:
                    alias_rows.append(
                        {
                            "entity_id": entity["entity_id"],
                            "display_name": entity["display_name"],
                            "alias": alias,
                            "source_scene_id": scene["scene_id"],
                        }
                    )
        for mystery in _as_list(discovery.get("mystery_entities")):
            mystery_entities.append(_normalize_mystery_entity(mystery, scene["scene_id"]))

    if config.dry_run:
        write_json(
            discovery_dir / "dry_run.json",
            {
                "volume_id": volume_meta["volume_id"],
                "model": config.model,
                "prompt_count": prompt_count,
                "message": "Prompts were generated; no Ollama requests were made.",
            },
        )
    else:
        write_jsonl(discovery_dir / "scene_discoveries.jsonl", scene_discoveries)
        write_jsonl(discovery_dir / "raw_responses.jsonl", raw_responses)
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
        "character_count": len(character_store.characters),
        "mystery_entity_count": len(mystery_entities),
    }
    write_json(discovery_dir / "run_summary.json", run_summary)
    return run_summary


def _build_scene_payload(
    scene: dict,
    paragraphs_by_id: dict[str, dict],
    dialogues: list[dict],
    known_characters: list[dict],
    volume_meta: dict,
) -> dict:
    paragraphs = [
        paragraphs_by_id[paragraph_id]
        for paragraph_id in scene["paragraph_ids"]
        if paragraph_id in paragraphs_by_id
    ]
    return {
        "volume_id": volume_meta["volume_id"],
        "scene": {
            "scene_id": scene["scene_id"],
            "chapter_id": scene["chapter_id"],
            "chapter_title": scene["chapter_title"],
        },
        "paragraphs": [
            {
                "paragraph_id": row["paragraph_id"],
                "text": row["text"],
            }
            for row in paragraphs
        ],
        "dialogues": [
            {
                "dialogue_id": row["dialogue_id"],
                "paragraph_id": row["paragraph_id"],
                "text": row["text"],
            }
            for row in dialogues
        ],
        "known_characters": known_characters,
    }


def _build_discovery_prompt(payload: dict) -> str:
    return (
        "你是轻小说说话人标注项目的阶段 1：角色发现与建库助手。\n"
        "你的任务不是给每句台词最终打标签，而是从当前场景中发现角色、别名、称谓、"
        "关系线索、说话风格线索、场景摘要，以及尚未命名但可追踪的神秘人物。\n\n"
        "规则：\n"
        "1. 只输出严格 JSON，不要 Markdown，不要解释性前后缀。\n"
        "2. 不要编造没有文本证据的人名。\n"
        "3. 如果人物未命名但看起来可在后文追踪，放入 mystery_entities。\n"
        "4. character_candidates 只放可以从文本中看到名字、称谓或稳定身份线索的人物。\n"
        "5. evidence 使用 paragraph_id 或 dialogue_id 相关的短证据，不要长篇复制原文。\n\n"
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


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _join_field(existing: str, new_value: str) -> str:
    if not new_value:
        return existing
    if not existing:
        return new_value
    if new_value in existing:
        return existing
    return f"{existing} / {new_value}"


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
