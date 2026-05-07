from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from novel_speaker_label.discovery import CharacterStore, _split_scene_into_requests
from novel_speaker_label.jsonl import read_jsonl
from novel_speaker_label.ollama_client import OllamaConfig, OllamaClient, collect_streaming_response
from novel_speaker_label.preprocess import PreprocessConfig, preprocess_volume


class PreprocessTests(unittest.TestCase):
    def test_preprocess_extracts_dialogues_with_context(self) -> None:
        source = "\n".join(
            [
                "目录",
                "序幕",
                "罗伦斯看着骑士。",
                "「要不要来一颗？」",
                "「呃……」",
                "骑士伸手取糖。",
                "第一幕",
                "赫萝笑了。",
                "「咱就是咱。」",
            ]
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            input_path = root / "volume01.txt"
            output_dir = root / "outputs" / "volume_01"
            input_path.write_text(source, encoding="utf-8")

            metadata = preprocess_volume(
                PreprocessConfig(
                    input_path=input_path,
                    output_dir=output_dir,
                    volume=1,
                    context_paragraphs=1,
                    max_scene_paragraphs=20,
                )
            )

            self.assertEqual(metadata["dialogue_count"], 3)
            dialogues = list(read_jsonl(output_dir / "preprocess" / "dialogues.jsonl"))
            self.assertEqual(dialogues[0]["text"], "要不要来一颗？")
            self.assertEqual(dialogues[1]["prev_context"][0]["text"], "「要不要来一颗？」")
            self.assertEqual(dialogues[2]["chapter_title"], "第一幕")


class DiscoveryRequestSplitTests(unittest.TestCase):
    def test_split_scene_limits_prompt_size(self) -> None:
        scene = {
            "scene_id": "volume_01-c001-s001",
            "chapter_id": "volume_01-c001",
            "chapter_title": "第一幕",
            "paragraph_ids": [f"p{i}" for i in range(5)],
        }
        paragraphs_by_id = {
            f"p{i}": {"paragraph_id": f"p{i}", "text": f"段落 {i}"}
            for i in range(5)
        }
        dialogues = [
            {"dialogue_id": f"d{i}", "paragraph_id": f"p{i}", "text": f"台词 {i}"}
            for i in range(5)
        ]

        jobs = _split_scene_into_requests(
            scene=scene,
            paragraphs_by_id=paragraphs_by_id,
            dialogues=dialogues,
            max_paragraphs=2,
            max_dialogues=2,
        )

        self.assertEqual(len(jobs), 3)
        self.assertLessEqual(max(len(job["paragraphs"]) for job in jobs), 2)
        self.assertLessEqual(max(len(job["dialogues"]) for job in jobs), 2)


class CharacterStoreTests(unittest.TestCase):
    def test_titles_do_not_merge_distinct_named_characters(self) -> None:
        store = CharacterStore()
        store.add_or_update(
            {"display_name": "罗伦斯", "aliases": ["老板"], "titles": ["旅行商人"]},
            "scene-1",
        )
        store.add_or_update(
            {"display_name": "杰廉", "titles": ["旅行商人", "年轻人"]},
            "scene-2",
        )

        self.assertEqual(
            [character["display_name"] for character in store.characters],
            ["罗伦斯", "杰廉"],
        )
        self.assertNotIn("老板", store.characters[0]["aliases"])
        self.assertIn("老板", store.characters[0]["titles"])

    def test_related_names_merge_without_generic_titles(self) -> None:
        store = CharacterStore()
        store.add_or_update(
            {"display_name": "列支敦·马贺特", "titles": ["行长"]},
            "scene-1",
        )
        store.add_or_update(
            {"display_name": "马贺特", "titles": ["行长"]},
            "scene-2",
        )

        self.assertEqual(len(store.characters), 1)
        self.assertEqual(store.characters[0]["display_name"], "列支敦·马贺特")


class OllamaClientTests(unittest.TestCase):
    def test_collect_streaming_response(self) -> None:
        lines = [
            b'{"response":"hello ","done":false}\n',
            b'{"response":"world","done":true}\n',
        ]

        self.assertEqual(collect_streaming_response(lines), "hello world")

    def test_collect_streaming_response_raises_on_ollama_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "model not found"):
            collect_streaming_response([b'{"error":"model not found"}\n'])

    def test_timeout_zero_disables_socket_timeout(self) -> None:
        client = OllamaClient(OllamaConfig(timeout=0))

        self.assertIsNone(client._socket_timeout())


if __name__ == "__main__":
    unittest.main()
