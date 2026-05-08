from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from novel_speaker_label.annotation import (
    AnnotationConfig,
    _aggregate_votes,
    _dialogue_windows,
    _extract_parsed_votes,
    _parse_json_response,
    _render_labeled_text,
    _select_mystery_candidates,
    annotate_volume,
)
from novel_speaker_label.discovery import CharacterStore, _split_scene_into_requests
from novel_speaker_label.jsonl import read_json, read_jsonl, write_json, write_jsonl
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
            self.assertEqual(dialogues[0]["dialogue_kind"], "standalone")
            self.assertEqual(dialogues[1]["prev_context"][0]["text"], "「要不要来一颗？」")
            self.assertEqual(dialogues[2]["chapter_title"], "第一幕")

    def test_preprocess_marks_inline_quotes(self) -> None:
        source = "\n".join(
            [
                "第一幕",
                "罗伦斯心想，如果乖乖说「是」的话，就不配当商人了。",
            ]
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            input_path = root / "volume01.txt"
            output_dir = root / "outputs" / "volume_01"
            input_path.write_text(source, encoding="utf-8")

            preprocess_volume(
                PreprocessConfig(
                    input_path=input_path,
                    output_dir=output_dir,
                    volume=1,
                    context_paragraphs=1,
                    max_scene_paragraphs=20,
                )
            )

            dialogues = list(read_jsonl(output_dir / "preprocess" / "dialogues.jsonl"))
            self.assertEqual(dialogues[0]["text"], "是")
            self.assertEqual(dialogues[0]["dialogue_kind"], "inline")


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


class AnnotationTests(unittest.TestCase):
    def test_single_model_confident_vote_with_anchor_is_accepted(self) -> None:
        dialogue = {
            "dialogue_id": "volume_01-d000001",
            "dialogue_index": 0,
            "volume_id": "volume_01",
            "chapter_id": "volume_01-c001",
            "chapter_index": 1,
            "chapter_title": "第一幕",
            "scene_id": "volume_01-c001-s001",
            "scene_index": 1,
            "paragraph_id": "p1",
            "paragraph_index": 0,
            "local_dialogue_index": 1,
            "text": "我是旅行商人罗伦斯。",
            "quote_text": "「我是旅行商人罗伦斯。」",
            "paragraph_text": "「我是旅行商人罗伦斯。」",
            "dialogue_kind": "standalone",
            "char_start": 0,
            "char_end": 12,
        }
        vote = {
            "dialogue_id": "volume_01-d000001",
            "paragraph_id": "p1",
            "char_start": 0,
            "char_end": 12,
            "model": "model-a",
            "weight": 1.0,
            "speaker_entity_id": "char_0001",
            "speaker_display": "罗伦斯",
            "speaker_status": "known",
            "confidence": 0.9,
            "candidate_speakers": [],
            "evidence": ["叙述提到罗伦斯开口"],
            "needs_review": False,
        }

        annotation = _aggregate_votes(
            dialogue, [vote], AnnotationConfig(output_dir=Path("unused"))
        )

        self.assertFalse(annotation["needs_review"])
        self.assertEqual(annotation["speaker_entity_id"], "char_0001")
        self.assertEqual(annotation["speaker_status"], "known")

    def test_single_model_known_vote_without_anchor_needs_review(self) -> None:
        dialogue = {
            "dialogue_id": "volume_01-d000002",
            "dialogue_index": 1,
            "volume_id": "volume_01",
            "chapter_id": "volume_01-c002",
            "chapter_index": 2,
            "chapter_title": "第一幕",
            "scene_id": "volume_01-c002-s001",
            "scene_index": 1,
            "paragraph_id": "p1",
            "paragraph_index": 0,
            "local_dialogue_index": 1,
            "text": "这是最后一件了吧？",
            "quote_text": "「这是最后一件了吧？」",
            "paragraph_text": "「这是最后一件了吧？」",
            "dialogue_kind": "standalone",
            "char_start": 0,
            "char_end": 11,
        }
        vote = {
            "dialogue_id": "volume_01-d000002",
            "paragraph_id": "p1",
            "char_start": 0,
            "char_end": 11,
            "model": "model-a",
            "weight": 1.0,
            "speaker_entity_id": "char_0001",
            "speaker_display": "罗伦斯",
            "speaker_status": "known",
            "confidence": 0.95,
            "candidate_speakers": [],
            "evidence": ["模型按交易轮次猜测"],
            "needs_review": False,
        }

        annotation = _aggregate_votes(
            dialogue, [vote], AnnotationConfig(output_dir=Path("unused"))
        )

        self.assertTrue(annotation["needs_review"])
        self.assertEqual(annotation["speaker_status"], "review")
        self.assertEqual(annotation["review_reason"], "single_model_without_anchor")

    def test_single_model_mystery_vote_needs_review(self) -> None:
        dialogue = {
            "dialogue_id": "volume_01-d000003",
            "dialogue_index": 2,
            "volume_id": "volume_01",
            "chapter_id": "volume_01-c002",
            "chapter_index": 2,
            "chapter_title": "第一幕",
            "scene_id": "volume_01-c002-s001",
            "scene_index": 1,
            "paragraph_id": "p1",
            "paragraph_index": 0,
            "local_dialogue_index": 1,
            "text": "嗯，这里确实有……七十件。多谢惠顾。",
            "quote_text": "「嗯，这里确实有……七十件。多谢惠顾。」",
            "paragraph_text": "「嗯，这里确实有……七十件。多谢惠顾。」",
            "dialogue_kind": "standalone",
            "char_start": 0,
            "char_end": 20,
        }
        vote = {
            "dialogue_id": "volume_01-d000003",
            "paragraph_id": "p1",
            "char_start": 0,
            "char_end": 20,
            "model": "model-a",
            "weight": 1.0,
            "speaker_entity_id": "mystery_volume_01_c002_s001_r001_修道院居民",
            "speaker_display": "修道院居民",
            "speaker_status": "mystery",
            "confidence": 0.9,
            "candidate_speakers": [],
            "evidence": ["模型把村民感谢误连到修道院居民"],
            "needs_review": False,
        }

        annotation = _aggregate_votes(
            dialogue, [vote], AnnotationConfig(output_dir=Path("unused"))
        )

        self.assertTrue(annotation["needs_review"])
        self.assertEqual(annotation["speaker_status"], "review")
        self.assertEqual(annotation["review_reason"], "single_model_mystery_candidate")

    def test_dialogue_windows_do_not_cross_scenes(self) -> None:
        dialogues = [
            {"dialogue_id": "d1", "scene_id": "s1"},
            {"dialogue_id": "d2", "scene_id": "s1"},
            {"dialogue_id": "d3", "scene_id": "s1"},
            {"dialogue_id": "d4", "scene_id": "s2"},
        ]

        windows = _dialogue_windows(dialogues, window_size=2)

        self.assertEqual(
            [[row["dialogue_id"] for row in window] for window in windows],
            [["d1", "d2"], ["d3"], ["d4"]],
        )

    def test_dialogue_windows_split_on_large_paragraph_gap(self) -> None:
        dialogues = [
            {"dialogue_id": "d1", "scene_id": "s1", "paragraph_index": 10},
            {"dialogue_id": "d2", "scene_id": "s1", "paragraph_index": 11},
            {"dialogue_id": "d3", "scene_id": "s1", "paragraph_index": 23},
        ]

        windows = _dialogue_windows(dialogues, window_size=8, max_paragraph_gap=4)

        self.assertEqual(
            [[row["dialogue_id"] for row in window] for window in windows],
            [["d1", "d2"], ["d3"]],
        )

    def test_extract_parsed_votes_reads_batch_response(self) -> None:
        parsed = _parse_json_response(
            '{"annotations":[{"dialogue_id":"d1","speaker_status":"known"},'
            '{"dialogue_id":"d2","speaker_status":"npc"}]}'
        )

        votes = _extract_parsed_votes(
            parsed,
            [{"dialogue_id": "d1"}, {"dialogue_id": "d2"}],
        )

        self.assertEqual(votes["d1"]["speaker_status"], "known")
        self.assertEqual(votes["d2"]["speaker_status"], "npc")

    def test_mystery_candidates_require_local_evidence_or_name(self) -> None:
        mysteries = [
            {
                "temporary_name": "修道院居民",
                "source_scene_id": "volume_01-c002-s001-r001",
                "evidence": ["volume_01-c002-s001-p000087"],
                "confidence": 0.8,
            },
            {
                "temporary_name": "小村落的村民",
                "source_scene_id": "volume_01-c002-s001-r003",
                "evidence": ["volume_01-c002-s001-p000081"],
                "confidence": 0.8,
            },
        ]

        selected = _select_mystery_candidates(
            mysteries=mysteries,
            context_ids={"volume_01-c002-s001-p000079", "volume_01-c002-s001-p000081"},
            context_text="只有罗伦斯先生你愿意到这深山里来，真是帮了我们大忙。",
            scene_id="volume_01-c002-s001",
            scene_memory_ids={"volume_01-c002-s001-r001"},
            limit=8,
        )

        self.assertEqual([row["temporary_name"] for row in selected], ["小村落的村民"])

    def test_render_labeled_text_inserts_review_label(self) -> None:
        paragraphs = [{"paragraph_id": "p1", "text": "「你好」「嗯」"}]
        annotations = [
            {
                "paragraph_id": "p1",
                "char_start": 0,
                "speaker_display": "罗伦斯",
                "speaker_status": "known",
                "needs_review": False,
            },
            {
                "paragraph_id": "p1",
                "char_start": 4,
                "speaker_display": "赫萝",
                "speaker_status": "review",
                "needs_review": True,
            },
        ]

        self.assertEqual(
            _render_labeled_text(paragraphs, annotations),
            "【罗伦斯】「你好」【待复核】「嗯」\n",
        )

    def test_render_labeled_text_skips_inline_quote_labels(self) -> None:
        paragraph = "如果乖乖说「是」的话，就不配当商人了。"
        paragraphs = [{"paragraph_id": "p1", "text": paragraph}]
        annotations = [
            {
                "paragraph_id": "p1",
                "char_start": paragraph.index("「"),
                "speaker_display": "骑士",
                "speaker_status": "known",
                "needs_review": False,
                "dialogue_kind": "inline",
            }
        ]

        self.assertEqual(_render_labeled_text(paragraphs, annotations), paragraph + "\n")

    def test_annotate_dry_run_writes_prompt_without_ollama(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output_dir = Path(tmp) / "outputs" / "volume_01"
            write_json(
                output_dir / "volume.json",
                {"volume_id": "volume_01", "volume": 1},
            )
            write_jsonl(
                output_dir / "preprocess" / "paragraphs.jsonl",
                [
                    {
                        "paragraph_id": "p1",
                        "text": "罗伦斯看着赫萝。「你好。」",
                    }
                ],
            )
            write_jsonl(
                output_dir / "preprocess" / "dialogues.jsonl",
                [
                    {
                        "dialogue_id": "volume_01-d000001",
                        "dialogue_index": 0,
                        "volume_id": "volume_01",
                        "chapter_id": "volume_01-c001",
                        "chapter_index": 1,
                        "chapter_title": "第一幕",
                        "scene_id": "volume_01-c001-s001",
                        "scene_index": 1,
                        "paragraph_id": "p1",
                        "paragraph_index": 0,
                        "local_dialogue_index": 1,
                        "text": "你好。",
                        "quote_text": "「你好。」",
                        "paragraph_text": "罗伦斯看着赫萝。「你好。」",
                        "char_start": 7,
                        "char_end": 12,
                        "prev_context": [],
                        "next_context": [],
                    }
                ],
            )
            write_jsonl(
                output_dir / "memory" / "semantic" / "characters.jsonl",
                [
                    {
                        "entity_id": "char_0001",
                        "display_name": "罗伦斯",
                        "aliases": [],
                        "titles": ["旅行商人"],
                        "description": "旅行商人",
                        "speech_style": "冷静",
                        "relationship_hints": [],
                        "confidence": 0.9,
                    }
                ],
            )
            write_jsonl(
                output_dir / "memory" / "episodic" / "scenes.jsonl",
                [
                    {
                        "scene_id": "volume_01-c001-s001-r001",
                        "parent_scene_id": "volume_01-c001-s001",
                        "scene_summary": "罗伦斯与赫萝交谈。",
                        "active_characters": ["罗伦斯", "赫萝"],
                        "relationships": [],
                        "chunk_index": 1,
                        "chunk_count": 1,
                    }
                ],
            )

            summary = annotate_volume(
                AnnotationConfig(output_dir=output_dir, models=("fake-model",), dry_run=True)
            )

            self.assertEqual(summary["prompt_count"], 1)
            self.assertEqual(summary["dialogue_count"], 1)
            dry_run = read_json(output_dir / "annotation" / "dry_run.json")
            self.assertEqual(dry_run["models"], ["fake-model"])
            self.assertTrue(
                (output_dir / "annotation" / "prompts" / "volume_01-d000001--fake_model.txt").exists()
            )
            prompt = (
                output_dir
                / "annotation"
                / "prompts"
                / "volume_01-d000001--fake_model.txt"
            ).read_text(encoding="utf-8")
            self.assertIn('"target_dialogue_ids"', prompt)
            self.assertIn('"annotations"', prompt)

    def test_annotate_cache_only_reads_batch_cache(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output_dir = Path(tmp) / "outputs" / "volume_01"
            write_json(
                output_dir / "volume.json",
                {"volume_id": "volume_01", "volume": 1},
            )
            write_jsonl(
                output_dir / "preprocess" / "paragraphs.jsonl",
                [
                    {"paragraph_id": "p1", "paragraph_index": 0, "text": "「你好。」"},
                    {"paragraph_id": "p2", "paragraph_index": 1, "text": "「嗯。」"},
                ],
            )
            base_dialogue = {
                "volume_id": "volume_01",
                "chapter_id": "volume_01-c001",
                "chapter_index": 1,
                "chapter_title": "第一幕",
                "scene_id": "volume_01-c001-s001",
                "scene_index": 1,
                "local_dialogue_index": 1,
                "char_start": 0,
                "char_end": 5,
                "prev_context": [],
                "next_context": [],
            }
            write_jsonl(
                output_dir / "preprocess" / "dialogues.jsonl",
                [
                    {
                        **base_dialogue,
                        "dialogue_id": "volume_01-d000001",
                        "dialogue_index": 0,
                        "paragraph_id": "p1",
                        "paragraph_index": 0,
                        "text": "你好。",
                        "quote_text": "「你好。」",
                        "paragraph_text": "「你好。」",
                    },
                    {
                        **base_dialogue,
                        "dialogue_id": "volume_01-d000002",
                        "dialogue_index": 1,
                        "paragraph_id": "p2",
                        "paragraph_index": 1,
                        "text": "嗯。",
                        "quote_text": "「嗯。」",
                        "paragraph_text": "「嗯。」",
                    },
                ],
            )
            write_jsonl(
                output_dir / "preprocess" / "scenes.jsonl",
                [
                    {
                        "scene_id": "volume_01-c001-s001",
                        "start_paragraph_index": 0,
                        "end_paragraph_index": 1,
                    }
                ],
            )
            write_json(
                output_dir
                / "annotation"
                / "cache"
                / "volume_01-d000001_to_volume_01-d000002--fake_model.json",
                {
                    "annotations": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "speaker_entity_id": "char_0001",
                            "speaker_display": "罗伦斯",
                            "speaker_status": "known",
                            "confidence": 0.9,
                        },
                        {
                            "dialogue_id": "volume_01-d000002",
                            "speaker_entity_id": "char_0002",
                            "speaker_display": "赫萝",
                            "speaker_status": "known",
                            "confidence": 0.9,
                        },
                    ]
                },
            )

            summary = annotate_volume(
                AnnotationConfig(
                    output_dir=output_dir,
                    models=("fake-model",),
                    cache_only=True,
                    write_prompts=False,
                    annotation_window_size=2,
                )
            )

            annotations = list(read_jsonl(output_dir / "annotation" / "annotations.jsonl"))
            self.assertEqual(summary["request_count"], 1)
            self.assertEqual(summary["vote_count"], 2)
            self.assertEqual([row["speaker_display"] for row in annotations], ["罗伦斯", "赫萝"])


if __name__ == "__main__":
    unittest.main()
