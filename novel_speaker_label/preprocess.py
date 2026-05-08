from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .jsonl import write_json, write_jsonl

CHAPTER_TITLE_RE = re.compile(
    r"^(序幕|终幕|尾声|后记|第[一二三四五六七八九十百零〇0-9]+[幕章话卷])$"
)
DIALOGUE_RE = re.compile(r"「([^」]*)」")
SCENE_BREAK_TITLES = {"插图"}
DECORATIVE_LINE_RE = re.compile(r"^[─\-—=＊*·\s]+$")


@dataclass(frozen=True)
class PreprocessConfig:
    input_path: Path
    output_dir: Path
    volume: int
    context_paragraphs: int = 3
    max_scene_paragraphs: int = 80
    copy_source: bool = True


def find_volume_file(novels_dir: Path, volume: int) -> Path:
    if not novels_dir.exists():
        raise FileNotFoundError(f"Novels directory does not exist: {novels_dir}")

    marker = f"{volume:02d}"
    candidates = sorted(
        path for path in novels_dir.iterdir() if path.is_file() and marker in path.name
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a novel file containing volume marker {marker!r} in {novels_dir}"
        )
    return candidates[0]


def preprocess_volume(config: PreprocessConfig) -> dict:
    raw_text = _read_text(config.input_path)
    lines = _normalize_lines(raw_text)
    volume_id = f"volume_{config.volume:02d}"

    paragraphs, chapters = _build_paragraphs_and_chapters(
        lines=lines,
        volume_id=volume_id,
        max_scene_paragraphs=config.max_scene_paragraphs,
    )
    dialogues = _extract_dialogues(
        paragraphs=paragraphs,
        volume_id=volume_id,
        context_paragraphs=config.context_paragraphs,
    )
    scenes = _build_scenes(paragraphs=paragraphs, dialogues=dialogues)

    preprocess_dir = config.output_dir / "preprocess"
    write_jsonl(preprocess_dir / "chapters.jsonl", chapters)
    write_jsonl(preprocess_dir / "paragraphs.jsonl", paragraphs)
    write_jsonl(preprocess_dir / "scenes.jsonl", scenes)
    write_jsonl(preprocess_dir / "dialogues.jsonl", dialogues)

    if config.copy_source:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.input_path, config.output_dir / "source.txt")

    metadata = {
        "volume_id": volume_id,
        "volume": config.volume,
        "source_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "chapter_count": len(chapters),
        "paragraph_count": len(paragraphs),
        "scene_count": len(scenes),
        "dialogue_count": len(dialogues),
        "context_paragraphs": config.context_paragraphs,
        "max_scene_paragraphs": config.max_scene_paragraphs,
        "files": {
            "chapters": str(preprocess_dir / "chapters.jsonl"),
            "paragraphs": str(preprocess_dir / "paragraphs.jsonl"),
            "scenes": str(preprocess_dir / "scenes.jsonl"),
            "dialogues": str(preprocess_dir / "dialogues.jsonl"),
        },
    }
    write_json(config.output_dir / "volume.json", metadata)
    return metadata


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8")


def _normalize_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    return [line.strip() for line in text.split("\n") if line.strip()]


def _build_paragraphs_and_chapters(
    lines: list[str], volume_id: str, max_scene_paragraphs: int
) -> tuple[list[dict], list[dict]]:
    paragraphs: list[dict] = []
    chapters: list[dict] = []

    current_chapter = {
        "chapter_id": f"{volume_id}-c000",
        "chapter_index": 0,
        "title": "front_matter",
        "start_paragraph_index": 0,
        "start_line_index": 0,
    }
    chapters.append(current_chapter)

    scene_index = 1
    scene_paragraph_count = 0
    previous_was_break = False

    for line_index, text in enumerate(lines):
        if _is_plain_chapter_title(text):
            if paragraphs:
                chapters[-1]["end_paragraph_index"] = len(paragraphs) - 1
            current_chapter = {
                "chapter_id": f"{volume_id}-c{len(chapters):03d}",
                "chapter_index": len(chapters),
                "title": text,
                "start_paragraph_index": len(paragraphs),
                "start_line_index": line_index,
            }
            chapters.append(current_chapter)
            scene_index = 1
            scene_paragraph_count = 0
            previous_was_break = False

        if _is_scene_break(text):
            previous_was_break = True
        elif previous_was_break:
            scene_index += 1
            scene_paragraph_count = 0
            previous_was_break = False
        elif scene_paragraph_count >= max_scene_paragraphs and not _looks_structural(text):
            scene_index += 1
            scene_paragraph_count = 0

        paragraph_id = (
            f"{current_chapter['chapter_id']}-s{scene_index:03d}-p{len(paragraphs) + 1:06d}"
        )
        paragraph = {
            "paragraph_id": paragraph_id,
            "volume_id": volume_id,
            "paragraph_index": len(paragraphs),
            "line_index": line_index,
            "chapter_id": current_chapter["chapter_id"],
            "chapter_index": current_chapter["chapter_index"],
            "chapter_title": current_chapter["title"],
            "scene_id": f"{current_chapter['chapter_id']}-s{scene_index:03d}",
            "scene_index": scene_index,
            "text": text,
            "is_structural": _looks_structural(text),
        }
        paragraphs.append(paragraph)
        scene_paragraph_count += 1

    if chapters:
        chapters[-1]["end_paragraph_index"] = len(paragraphs) - 1
    return paragraphs, chapters


def _is_plain_chapter_title(text: str) -> bool:
    return bool(CHAPTER_TITLE_RE.match(text))


def _is_scene_break(text: str) -> bool:
    return text in SCENE_BREAK_TITLES or bool(DECORATIVE_LINE_RE.match(text))


def _looks_structural(text: str) -> bool:
    if _is_scene_break(text):
        return True
    if _is_plain_chapter_title(text):
        return True
    if len(text) <= 2 and text in {"序", "幕", "终", "第"}:
        return True
    return False


def _extract_dialogues(
    paragraphs: list[dict], volume_id: str, context_paragraphs: int
) -> list[dict]:
    dialogues: list[dict] = []
    dialogue_index = 0

    for paragraph in paragraphs:
        text = paragraph["text"]
        matches = list(DIALOGUE_RE.finditer(text))
        for local_index, match in enumerate(matches, start=1):
            dialogue_index += 1
            dialogue_id = (
                f"{volume_id}-d{dialogue_index:06d}"
            )
            paragraph_index = paragraph["paragraph_index"]
            prev_context = _paragraph_context(
                paragraphs, paragraph_index - context_paragraphs, paragraph_index
            )
            next_context = _paragraph_context(
                paragraphs,
                paragraph_index + 1,
                paragraph_index + context_paragraphs + 1,
            )
            dialogues.append(
                {
                    "dialogue_id": dialogue_id,
                    "dialogue_index": dialogue_index - 1,
                    "volume_id": volume_id,
                    "chapter_id": paragraph["chapter_id"],
                    "chapter_index": paragraph["chapter_index"],
                    "chapter_title": paragraph["chapter_title"],
                    "scene_id": paragraph["scene_id"],
                    "scene_index": paragraph["scene_index"],
                    "paragraph_id": paragraph["paragraph_id"],
                    "paragraph_index": paragraph["paragraph_index"],
                    "local_dialogue_index": local_index,
                    "text": match.group(1).strip(),
                    "quote_text": match.group(0),
                    "paragraph_text": text,
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "prev_context": prev_context,
                    "next_context": next_context,
                }
            )
    return dialogues


def _paragraph_context(paragraphs: list[dict], start: int, stop: int) -> list[dict]:
    rows = []
    for paragraph in paragraphs[max(0, start) : min(len(paragraphs), stop)]:
        rows.append(
            {
                "paragraph_id": paragraph["paragraph_id"],
                "chapter_id": paragraph["chapter_id"],
                "scene_id": paragraph["scene_id"],
                "text": paragraph["text"],
            }
        )
    return rows


def _build_scenes(paragraphs: list[dict], dialogues: list[dict]) -> list[dict]:
    scenes_by_id: dict[str, dict] = {}
    for paragraph in paragraphs:
        scene_id = paragraph["scene_id"]
        scene = scenes_by_id.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "volume_id": paragraph["volume_id"],
                "chapter_id": paragraph["chapter_id"],
                "chapter_index": paragraph["chapter_index"],
                "chapter_title": paragraph["chapter_title"],
                "scene_index": paragraph["scene_index"],
                "start_paragraph_index": paragraph["paragraph_index"],
                "end_paragraph_index": paragraph["paragraph_index"],
                "paragraph_ids": [],
                "dialogue_ids": [],
            },
        )
        scene["end_paragraph_index"] = paragraph["paragraph_index"]
        scene["paragraph_ids"].append(paragraph["paragraph_id"])

    for dialogue in dialogues:
        scene = scenes_by_id[dialogue["scene_id"]]
        scene["dialogue_ids"].append(dialogue["dialogue_id"])

    return list(scenes_by_id.values())
