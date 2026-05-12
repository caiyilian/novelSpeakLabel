from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from novel_speaker_label.cli import _resolve_output
from novel_speaker_label.annotation import (
    AnnotationConfig,
    _aggregate_votes,
    _apply_contradiction_checks,
    _build_judge_prompt,
    _dialogue_windows,
    _extract_parsed_votes,
    _normalize_contradiction_check,
    _normalize_vote,
    _parse_json_response,
    _render_labeled_text,
    _rule_votes_for_window,
    _select_mystery_candidates,
    annotate_volume,
)
from novel_speaker_label.discovery import CharacterStore, _split_scene_into_requests
from novel_speaker_label.jsonl import read_json, read_jsonl, write_json, write_jsonl
from novel_speaker_label.ollama_client import OllamaConfig, OllamaClient, collect_streaming_response
from novel_speaker_label.preprocess import PreprocessConfig, preprocess_volume
from novel_speaker_label.reading_v2 import (
    ReadingV2Config,
    annotate_v2_volume,
    estimate_prompt_tokens,
    prompt_length_status,
)


def _annotation_dialogue(
    dialogue_id: str,
    dialogue_index: int,
    paragraph_index: int,
    text: str,
    *,
    paragraph_text: str | None = None,
    prev_context: list[dict] | None = None,
    next_context: list[dict] | None = None,
) -> dict:
    quote_text = f"「{text}」"
    full_paragraph = paragraph_text or quote_text
    return {
        "dialogue_id": dialogue_id,
        "dialogue_index": dialogue_index,
        "volume_id": "volume_01",
        "chapter_id": "volume_01-c002",
        "chapter_index": 2,
        "chapter_title": "第一幕",
        "scene_id": "volume_01-c002-s001",
        "scene_index": 1,
        "paragraph_id": f"p{paragraph_index}",
        "paragraph_index": paragraph_index,
        "local_dialogue_index": 1,
        "text": text,
        "quote_text": quote_text,
        "paragraph_text": full_paragraph,
        "dialogue_kind": "standalone",
        "char_start": 0,
        "char_end": len(full_paragraph),
        "prev_context": prev_context or [],
        "next_context": next_context or [],
    }


def _annotation_vote(
    dialogue: dict,
    entity_id: str,
    display: str,
    *,
    model: str = "model-a",
    confidence: float = 0.9,
) -> dict:
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "paragraph_id": dialogue["paragraph_id"],
        "char_start": dialogue["char_start"],
        "char_end": dialogue["char_end"],
        "model": model,
        "weight": 1.0,
        "speaker_entity_id": entity_id,
        "speaker_display": display,
        "speaker_status": "known",
        "confidence": confidence,
        "candidate_speakers": [],
        "evidence": ["测试投票"],
        "needs_review": False,
    }


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

    def test_speech_markers_are_stored_separately_from_titles(self) -> None:
        store = CharacterStore()
        store.add_or_update(
            {
                "display_name": "阿晴",
                "titles": ["巫女"],
                "speech_markers": ["妾身", "也罢"],
            },
            "scene-1",
        )
        store.add_or_update(
            {
                "display_name": "阿晴",
                "speech_markers": ["妾身", "罢了"],
            },
            "scene-2",
        )

        self.assertEqual(store.characters[0]["speech_markers"], ["妾身", "也罢", "罢了"])
        self.assertNotIn("妾身", store.characters[0]["titles"])
        self.assertEqual(
            store.snapshot_for_prompt(1)[0]["speech_markers"],
            ["妾身", "也罢", "罢了"],
        )

    def test_sentence_like_speech_markers_are_filtered(self) -> None:
        store = CharacterStore()
        store.add_or_update(
            {
                "display_name": "阿晴",
                "speech_markers": ["妾身", "你到底是谁？", "好名字呗？", "也罢"],
            },
            "scene-1",
        )

        self.assertEqual(store.characters[0]["speech_markers"], ["妾身", "也罢"])


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

    def test_structured_single_model_known_vote_without_anchor_is_accepted(self) -> None:
        dialogue = _annotation_dialogue(
            "volume_01-d000002",
            1,
            0,
            "这是最后一件了吧？",
        )
        vote = _annotation_vote(
            dialogue,
            "char_0001",
            "罗伦斯",
            confidence=0.95,
        )

        annotation = _aggregate_votes(
            dialogue,
            [vote],
            AnnotationConfig(output_dir=Path("unused")),
            structured=True,
        )

        self.assertFalse(annotation["needs_review"])
        self.assertEqual(annotation["speaker_status"], "known")

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

    def test_group_npc_vote_is_normalized_from_mystery(self) -> None:
        dialogue = _annotation_dialogue("volume_01-d000010", 9, 10, "谢谢惠顾。")

        vote = _normalize_vote(
            parsed_vote={
                "dialogue_id": dialogue["dialogue_id"],
                "speaker_entity_id": "mystery_village_group",
                "speaker_display": "村民群体",
                "speaker_status": "mystery",
                "confidence": 0.88,
                "needs_review": False,
            },
            dialogue=dialogue,
            model="judge:model-a",
            weight=1.0,
        )

        self.assertEqual(vote["speaker_status"], "npc")
        self.assertEqual(vote["speaker_entity_id"], "npc:村民群体")
        self.assertFalse(vote["needs_review"])

    def test_rule_votes_use_addressed_name_and_turns_for_village_trade(self) -> None:
        payload = {
            "candidate_characters": [
                {
                    "entity_id": "char_0001",
                    "display_name": "罗伦斯",
                    "aliases": ["罗伦斯先生"],
                }
            ]
        }
        dialogues = [
            _annotation_dialogue("volume_01-d000002", 1, 78, "这是最后一件了吧？"),
            _annotation_dialogue(
                "volume_01-d000003",
                2,
                79,
                "嗯，这里确实有……七十件。多谢惠顾。",
            ),
            _annotation_dialogue(
                "volume_01-d000004",
                3,
                80,
                "不，我们才要谢谢你呢。只有罗伦斯先生你愿意到这深山里来，真是帮了我们大忙。",
            ),
            _annotation_dialogue(
                "volume_01-d000005",
                4,
                81,
                "不过，我也因此拿到上等的皮草啊，我会再来的。",
            ),
        ]

        votes = _rule_votes_for_window(dialogues, payload)
        by_id = {vote["dialogue_id"]: vote for vote in votes}

        self.assertEqual(by_id["volume_01-d000002"]["speaker_display"], "小村落的村民")
        self.assertEqual(by_id["volume_01-d000003"]["speaker_display"], "罗伦斯")
        self.assertEqual(by_id["volume_01-d000004"]["speaker_display"], "小村落的村民")
        self.assertEqual(by_id["volume_01-d000005"]["speaker_display"], "罗伦斯")

    def test_rule_votes_do_not_let_direct_attribution_cross_dialogues(self) -> None:
        payload = {
            "candidate_characters": [
                {"entity_id": "char_0001", "display_name": "罗伦斯", "aliases": []},
                {"entity_id": "char_0003", "display_name": "骑士", "aliases": []},
            ]
        }
        dialogues = [
            _annotation_dialogue(
                "volume_01-d000014",
                13,
                112,
                "原来如此。我看你车上还有货，是盐吗？",
                next_context=[
                    {"text": "「不，这些是皮草。您瞧。」"},
                    {"text": "罗伦斯一边说，一边转向货台掀开覆盖的麻布。那是非常漂亮的貂皮。如果以眼前这位骑士的薪水来看，相信貂皮的价值比他的年薪还要高。"},
                ],
            ),
            _annotation_dialogue(
                "volume_01-d000015",
                14,
                113,
                "不，这些是皮草。您瞧。",
                next_context=[
                    {"text": "罗伦斯一边说，一边转向货台掀开覆盖的麻布。那是非常漂亮的貂皮。如果以眼前这位骑士的薪水来看，相信貂皮的价值比他的年薪还要高。"}
                ],
            ),
            _annotation_dialogue(
                "volume_01-d000016",
                15,
                115,
                "喔，那这是什么？",
                prev_context=[
                    {"text": "「原来如此。我看你车上还有货，是盐吗？」"},
                    {"text": "「不，这些是皮草。您瞧。」"},
                    {"text": "罗伦斯一边说，一边转向货台掀开覆盖的麻布。那是非常漂亮的貂皮。如果以眼前这位骑士的薪水来看，相信貂皮的价值比他的年薪还要高。"},
                ],
            ),
            _annotation_dialogue(
                "volume_01-d000017",
                16,
                116,
                "啊，这是山里的村民给我的麦子。",
            ),
            _annotation_dialogue("volume_01-d000018", 17, 118, "嗯。好了，你可以走了。"),
        ]

        votes = _rule_votes_for_window(dialogues, payload)
        by_id = {vote["dialogue_id"]: vote for vote in votes}

        self.assertEqual(by_id["volume_01-d000014"]["speaker_display"], "骑士")
        self.assertEqual(by_id["volume_01-d000015"]["speaker_display"], "罗伦斯")
        self.assertEqual(by_id["volume_01-d000016"]["speaker_display"], "骑士")
        self.assertEqual(by_id["volume_01-d000017"]["speaker_display"], "罗伦斯")
        self.assertEqual(by_id["volume_01-d000018"]["speaker_display"], "骑士")

    def test_rule_votes_use_farm_greeting_anchor_and_turns(self) -> None:
        payload = {
            "candidate_characters": [
                {"entity_id": "char_0001", "display_name": "罗伦斯", "aliases": []},
                {"entity_id": "char_0005", "display_name": "农夫", "aliases": []},
            ]
        }
        dialogues = [
            _annotation_dialogue(
                "volume_01-d000027",
                26,
                149,
                "嗨！辛苦了。",
                next_context=[
                    {"text": "罗伦斯朝正在帕斯罗村的麦田一角，把麦子往马车上堆的农夫打招呼。"}
                ],
            ),
            _annotation_dialogue("volume_01-d000028", 27, 151, "喔？"),
            _annotation_dialogue("volume_01-d000029", 28, 152, "请问叶勒在哪里啊？"),
            _annotation_dialogue(
                "volume_01-d000030",
                29,
                153,
                "喔！叶勒在那儿。",
                next_context=[{"text": "农夫晒得黝黑的脸上堆满了笑容说道。"}],
            ),
        ]

        votes = _rule_votes_for_window(dialogues, payload)
        by_id = {vote["dialogue_id"]: vote for vote in votes}

        self.assertEqual(by_id["volume_01-d000027"]["speaker_display"], "罗伦斯")
        self.assertEqual(by_id["volume_01-d000028"]["speaker_display"], "农夫")
        self.assertEqual(by_id["volume_01-d000029"]["speaker_display"], "罗伦斯")
        self.assertEqual(by_id["volume_01-d000030"]["speaker_display"], "农夫")

    def test_rule_vote_overrides_wrong_model_majority(self) -> None:
        payload = {
            "candidate_characters": [
                {"entity_id": "char_0001", "display_name": "罗伦斯", "aliases": []},
                {"entity_id": "char_0003", "display_name": "骑士", "aliases": []},
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000020",
            19,
            120,
            "发生什么事了吗？平常在这里应该见不到骑士吧？",
            prev_context=[
                {"text": "罗伦斯一边有意无意地把玩刚刚的皮袋，一边转回骑士的方向。"}
            ],
            next_context=[
                {"text": "年轻骑士可能是因为被询问而感到不悦，稍稍皱起眉头，再看到罗伦斯手中的皮袋，眉头皱得更深了。"}
            ],
        )
        rule_vote = _rule_votes_for_window([dialogue], payload)[0]
        model_votes = [
            _annotation_vote(dialogue, "char_0003", "骑士", model="model-a"),
            _annotation_vote(dialogue, "char_0003", "骑士", model="model-b"),
            _annotation_vote(dialogue, "char_0003", "骑士", model="model-c"),
        ]

        annotation = _aggregate_votes(
            dialogue,
            [*model_votes, rule_vote],
            AnnotationConfig(output_dir=Path("unused")),
        )

        self.assertFalse(annotation["needs_review"])
        self.assertTrue(annotation["rule_applied"])
        self.assertEqual(annotation["speaker_display"], "罗伦斯")

    def test_reported_speech_reference_is_not_a_single_model_anchor(self) -> None:
        dialogue = _annotation_dialogue(
            "volume_01-d000025",
            24,
            132,
            "异教徒祭典……还真会猜啊。",
            next_context=[
                {"text": "离开修道院一会儿后，罗伦斯喃喃念着骑士说的话，苦笑了一下。"}
            ],
        )
        vote = _annotation_vote(dialogue, "char_0003", "骑士", confidence=0.95)

        annotation = _aggregate_votes(
            dialogue, [vote], AnnotationConfig(output_dir=Path("unused"))
        )

        self.assertTrue(annotation["needs_review"])
        self.assertEqual(annotation["review_reason"], "single_model_without_anchor")

    def test_rule_votes_do_not_use_later_inline_reply_as_current_attribution(self) -> None:
        payload = {
            "candidate_characters": [
                {"entity_id": "char_0001", "display_name": "罗伦斯", "aliases": []},
                {"entity_id": "char_0003", "display_name": "骑士", "aliases": []},
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000022",
            21,
            126,
            "听说最近，这一带会有异教徒祭典，所以我们才受命在这里防卫。你知道什么消息吗？",
            next_context=[
                {
                    "text": "这时，如果表现出失望的表情，那么演技就太差了。罗伦斯假装想了好一会儿后，回答说：「我不知道耶。」事实上，罗伦斯是在撒谎。"
                }
            ],
        )

        votes = _rule_votes_for_window([dialogue], payload)

        self.assertEqual(votes, [])

    def test_rule_votes_do_not_treat_indirect_speech_as_direct_attribution(self) -> None:
        payload = {
            "candidate_characters": [
                {"entity_id": "char_0001", "display_name": "罗伦斯", "aliases": []},
                {"entity_id": "char_0002", "display_name": "赫萝", "aliases": []},
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000114",
            113,
            365,
            "咱变身需要的东西是一些麦子。",
            next_context=[
                {"text": "麦子听起来挺像是丰收之神会要的报偿，罗伦斯似乎能够理解这说法。"}
            ],
        )

        votes = _rule_votes_for_window([dialogue], payload)

        self.assertEqual(votes, [])

    def test_speech_marker_contradiction_uses_character_memory(self) -> None:
        dialogue = _annotation_dialogue(
            "volume_01-d000146",
            145,
            427,
            "妾身真是佩服你。",
        )
        votes = [
            _annotation_vote(dialogue, "char_0001", "罗伦斯", model="model-a"),
            _annotation_vote(dialogue, "char_0001", "罗伦斯", model="model-b"),
        ]
        speaker_options = [
            {
                "entity_id": "char_0001",
                "display": "罗伦斯",
                "status": "known",
                "names": ["罗伦斯"],
                "speech_markers": [],
            },
            {
                "entity_id": "char_0002",
                "display": "阿晴",
                "status": "known",
                "names": ["阿晴"],
                "speech_markers": ["妾身"],
            },
        ]

        annotation = _aggregate_votes(
            dialogue,
            votes,
            AnnotationConfig(output_dir=Path("unused")),
            speaker_options=speaker_options,
        )

        self.assertTrue(annotation["needs_review"])
        self.assertEqual(annotation["review_reason"], "speaker_contradiction:speech_marker:阿晴")

    def test_speech_marker_rule_votes_use_character_memory(self) -> None:
        payload = {
            "candidate_characters": [
                {
                    "entity_id": "char_0001",
                    "display_name": "阿晴",
                    "aliases": [],
                    "speech_markers": ["妾身"],
                },
                {
                    "entity_id": "char_0002",
                    "display_name": "罗伦斯",
                    "aliases": [],
                    "speech_markers": [],
                },
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000146",
            145,
            427,
            "妾身真是佩服你。",
        )

        votes = _rule_votes_for_window([dialogue], payload)

        self.assertEqual(len(votes), 1)
        self.assertEqual(votes[0]["speaker_display"], "阿晴")

    def test_noisy_speech_marker_is_ignored(self) -> None:
        payload = {
            "candidate_characters": [
                {
                    "entity_id": "char_0001",
                    "display_name": "罗伦斯",
                    "aliases": [],
                    "speech_markers": ["……"],
                }
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000147",
            146,
            428,
            "……这是怎么回事？",
        )

        self.assertEqual(_rule_votes_for_window([dialogue], payload), [])

    def test_legacy_title_speech_marker_votes_are_supported(self) -> None:
        payload = {
            "candidate_characters": [
                {
                    "entity_id": "char_0001",
                    "display_name": "阿晴",
                    "aliases": [],
                    "titles": ["妾身"],
                },
                {
                    "entity_id": "char_0002",
                    "display_name": "罗伦斯",
                    "aliases": [],
                    "titles": [],
                },
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000146",
            145,
            427,
            "妾身真是佩服你。",
        )

        votes = _rule_votes_for_window([dialogue], payload)

        self.assertEqual(len(votes), 1)
        self.assertEqual(votes[0]["speaker_display"], "阿晴")

    def test_third_person_name_reference_is_not_self_intro(self) -> None:
        payload = {
            "candidate_characters": [
                {"entity_id": "char_0001", "display_name": "罗伦斯", "aliases": []},
                {"entity_id": "char_0002", "display_name": "阿晴", "aliases": []},
            ]
        }
        dialogue = _annotation_dialogue(
            "volume_01-d000952",
            951,
            2065,
            "谢谢。她应该……不，她一定会来的！她的名字是阿晴，是个头上套着外套，身材娇小的女孩。",
        )
        votes = [
            _annotation_vote(dialogue, "char_0002", "阿晴", model="model-a"),
            _annotation_vote(dialogue, "char_0002", "阿晴", model="model-b"),
        ]
        speaker_options = [
            {
                "entity_id": "char_0001",
                "display": "罗伦斯",
                "status": "known",
                "names": ["罗伦斯"],
                "speech_markers": [],
            },
            {
                "entity_id": "char_0002",
                "display": "阿晴",
                "status": "known",
                "names": ["阿晴"],
                "speech_markers": [],
            },
        ]

        self.assertEqual(_rule_votes_for_window([dialogue], payload), [])
        annotation = _aggregate_votes(
            dialogue,
            votes,
            AnnotationConfig(output_dir=Path("unused")),
            speaker_options=speaker_options,
        )
        self.assertTrue(annotation["needs_review"])
        self.assertEqual(
            annotation["review_reason"],
            "speaker_contradiction:third_person_name_reference",
        )

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

    def test_structured_annotate_cache_only_reads_role_caches(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output_dir = Path(tmp) / "outputs2" / "volume_01"
            write_json(
                output_dir / "volume.json",
                {"volume_id": "volume_01", "volume": 1},
            )
            paragraph = "罗伦斯说：「你好。」"
            write_jsonl(
                output_dir / "preprocess" / "paragraphs.jsonl",
                [{"paragraph_id": "p1", "paragraph_index": 0, "text": paragraph}],
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
                        "dialogue_kind": "standalone",
                        "paragraph_text": paragraph,
                        "char_start": paragraph.index("「"),
                        "char_end": len(paragraph),
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
                        "speech_markers": [],
                        "relationship_hints": [],
                        "confidence": 0.9,
                    }
                ],
            )

            write_json(
                output_dir
                / "annotation"
                / "evidence_cache"
                / "evidence--volume_01-d000001--fake_model.json",
                {
                    "evidence": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "explicit_attribution": {
                                "speaker_display": "罗伦斯",
                                "text": "罗伦斯说",
                            },
                            "positive_evidence": ["叙述明确罗伦斯说话"],
                        }
                    ]
                },
            )
            write_json(
                output_dir
                / "annotation"
                / "judgement_cache"
                / "judge--volume_01-d000001--fake_model.json",
                {
                    "annotations": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "speaker_entity_id": "char_0001",
                            "speaker_display": "罗伦斯",
                            "speaker_status": "known",
                            "confidence": 0.9,
                            "evidence": ["叙述明确罗伦斯说"],
                        }
                    ]
                },
            )
            write_json(
                output_dir
                / "annotation"
                / "contradiction_cache"
                / "contradiction--volume_01-d000001--fake_model.json",
                {
                    "checks": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "has_contradiction": False,
                            "severity": "none",
                            "reason": "",
                            "needs_review": False,
                        }
                    ]
                },
            )

            summary = annotate_volume(
                AnnotationConfig(
                    output_dir=output_dir,
                    models=("fake-model",),
                    pipeline="structured",
                    cache_only=True,
                    write_prompts=False,
                )
            )

            annotations = list(read_jsonl(output_dir / "annotation" / "annotations.jsonl"))
            self.assertEqual(summary["pipeline"], "structured")
            self.assertEqual(summary["prompt_count"], 3)
            self.assertEqual(summary["evidence_count"], 1)
            self.assertEqual(summary["judgement_count"], 1)
            self.assertEqual(summary["contradiction_check_count"], 1)
            self.assertEqual(annotations[0]["speaker_display"], "罗伦斯")
            self.assertFalse(annotations[0]["needs_review"])
            self.assertTrue((output_dir / "annotation" / "evidence.jsonl").exists())
            self.assertTrue((output_dir / "annotation" / "judgements.jsonl").exists())
            self.assertTrue(
                (output_dir / "annotation" / "contradiction_checks.jsonl").exists()
            )

    def test_structured_repair_cache_resolves_strong_contradiction(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output_dir = Path(tmp) / "outputs2" / "volume_01"
            write_json(
                output_dir / "volume.json",
                {"volume_id": "volume_01", "volume": 1},
            )
            paragraph = "罗伦斯看着骑士。「请问叶勒在哪里？」"
            char_start = paragraph.index("「")
            write_jsonl(
                output_dir / "preprocess" / "paragraphs.jsonl",
                [{"paragraph_id": "p1", "paragraph_index": 0, "text": paragraph}],
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
                        "chapter_title": "第一章",
                        "scene_id": "volume_01-c001-s001",
                        "scene_index": 1,
                        "paragraph_id": "p1",
                        "paragraph_index": 0,
                        "local_dialogue_index": 1,
                        "text": "请问叶勒在哪里？",
                        "quote_text": "「请问叶勒在哪里？」",
                        "dialogue_kind": "standalone",
                        "paragraph_text": paragraph,
                        "char_start": char_start,
                        "char_end": len(paragraph),
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
                        "titles": [],
                        "description": "旅行商人",
                        "speech_markers": [],
                        "relationship_hints": [],
                        "confidence": 0.9,
                    },
                    {
                        "entity_id": "char_0002",
                        "display_name": "骑士",
                        "aliases": [],
                        "titles": [],
                        "description": "守卫骑士",
                        "speech_markers": [],
                        "relationship_hints": [],
                        "confidence": 0.9,
                    },
                ],
            )

            write_json(
                output_dir
                / "annotation"
                / "evidence_cache"
                / "evidence--volume_01-d000001--fake_model.json",
                {
                    "evidence": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "positive_evidence": ["第一轮有邻近上下文证据"],
                            "negative_evidence": [],
                        }
                    ]
                },
            )
            write_json(
                output_dir
                / "annotation"
                / "judgement_cache"
                / "judge--volume_01-d000001--fake_model.json",
                {
                    "annotations": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "speaker_entity_id": "char_0001",
                            "speaker_display": "罗伦斯",
                            "speaker_status": "known",
                            "confidence": 0.9,
                            "evidence": ["第一轮误判为罗伦斯"],
                        }
                    ]
                },
            )
            write_json(
                output_dir
                / "annotation"
                / "contradiction_cache"
                / "contradiction--volume_01-d000001--fake_model.json",
                {
                    "checks": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "has_contradiction": True,
                            "severity": "strong",
                            "reason": "同段叙述明确骑士在回答",
                            "counter_evidence": ["骑士回答"],
                            "needs_review": True,
                        }
                    ]
                },
            )
            write_json(
                output_dir
                / "annotation"
                / "repair_cache"
                / "repair--volume_01-d000001--iter1--fake_model.json",
                {
                    "annotations": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "speaker_entity_id": "char_0002",
                            "speaker_display": "骑士",
                            "speaker_status": "known",
                            "confidence": 0.92,
                            "evidence": ["反证指向骑士回答"],
                            "needs_review": False,
                        }
                    ]
                },
            )

            summary = annotate_volume(
                AnnotationConfig(
                    output_dir=output_dir,
                    models=("fake-model",),
                    pipeline="structured",
                    cache_only=True,
                    write_prompts=False,
                )
            )

            annotations = list(read_jsonl(output_dir / "annotation" / "annotations.jsonl"))
            repairs = list(read_jsonl(output_dir / "annotation" / "repairs.jsonl"))
            self.assertEqual(summary["prompt_count"], 4)
            self.assertEqual(summary["repair_count"], 1)
            self.assertEqual(annotations[0]["speaker_display"], "骑士")
            self.assertEqual(annotations[0]["speaker_status"], "known")
            self.assertFalse(annotations[0]["needs_review"])
            self.assertEqual(annotations[0]["repair_trace"]["stop_reason"], "resolved")
            self.assertEqual(annotations[0]["repair_trace"]["iteration_count"], 1)
            self.assertEqual(annotations[0]["contradiction_checks"][0]["severity"], "strong")
            self.assertTrue(repairs[0]["resolved"])

    def test_weak_contradiction_does_not_force_review(self) -> None:
        dialogue = _annotation_dialogue("d1", 0, 0, "她的名字是赫萝。")
        annotation = _annotation_vote(dialogue, "char_0001", "罗伦斯")
        check = _normalize_contradiction_check(
            {
                "dialogue_id": "d1",
                "has_contradiction": False,
                "severity": "weak",
                "reason": "台词提到赫萝，但不是强反证",
                "counter_evidence": ["台词提到赫萝"],
                "needs_review": True,
            },
            dialogue,
            "checker",
            "request-1",
        )

        updated = _apply_contradiction_checks([annotation], [check])

        self.assertFalse(check["needs_review"])
        self.assertEqual(check["severity"], "weak")
        self.assertFalse(updated[0]["needs_review"])
        self.assertEqual(updated[0]["speaker_status"], "known")
        self.assertNotIn("review_reason", updated[0])
        self.assertEqual(updated[0]["contradiction_checks"][0]["severity"], "weak")

    def test_strong_contradiction_forces_review(self) -> None:
        dialogue = _annotation_dialogue("d1", 0, 0, "咱就是咱。")
        annotation = _annotation_vote(dialogue, "char_0001", "罗伦斯")
        check = _normalize_contradiction_check(
            {
                "dialogue_id": "d1",
                "has_contradiction": False,
                "severity": "strong",
                "reason": "命中唯一属于赫萝的口癖",
                "counter_evidence": ["咱"],
                "needs_review": False,
            },
            dialogue,
            "checker",
            "request-1",
        )

        updated = _apply_contradiction_checks([annotation], [check])

        self.assertTrue(check["needs_review"])
        self.assertTrue(updated[0]["needs_review"])
        self.assertEqual(updated[0]["speaker_status"], "review")
        self.assertEqual(
            updated[0]["review_reason"],
            "model_contradiction:命中唯一属于赫萝的口癖",
        )

    def test_structured_judge_prompt_requests_compact_output(self) -> None:
        prompt = _build_judge_prompt(
            {
                "target_dialogue_ids": ["d1"],
                "dialogues": [],
                "candidate_characters": [],
                "structured_evidence": [],
            }
        )

        self.assertIn("candidate_speakers 最多 2 个候选", prompt)
        self.assertIn("evidence 最多 3 条", prompt)
        self.assertNotIn('"should_create_new_entity": false', prompt)


class ReadingV2Tests(unittest.TestCase):
    def _write_v2_fixture(self, output_dir: Path) -> None:
        paragraph = "「Hello there.」"
        write_json(output_dir / "volume.json", {"volume_id": "volume_01", "volume": 1})
        write_jsonl(
            output_dir / "preprocess" / "paragraphs.jsonl",
            [
                {
                    "paragraph_id": "p1",
                    "paragraph_index": 0,
                    "volume_id": "volume_01",
                    "chapter_id": "volume_01-c001",
                    "chapter_index": 1,
                    "chapter_title": "Chapter 1",
                    "scene_id": "volume_01-c001-s001",
                    "scene_index": 1,
                    "text": paragraph,
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
                    "chapter_title": "Chapter 1",
                    "scene_id": "volume_01-c001-s001",
                    "scene_index": 1,
                    "paragraph_id": "p1",
                    "paragraph_index": 0,
                    "local_dialogue_index": 1,
                    "text": "Hello there.",
                    "quote_text": paragraph,
                    "dialogue_kind": "standalone",
                    "char_start": 0,
                    "char_end": len(paragraph),
                }
            ],
        )

    def test_prompt_length_status_uses_hard_limit(self) -> None:
        self.assertGreater(estimate_prompt_tokens("汉字abc{}"), 0)
        self.assertEqual(prompt_length_status(8, 10, 20), "ok")
        self.assertEqual(prompt_length_status(15, 10, 20), "near_limit")
        self.assertEqual(prompt_length_status(25, 10, 20), "over_limit")

    def test_annotate_v2_dry_run_writes_prompt_length_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output_dir = Path(tmp) / "outputs" / "volume_01"
            self._write_v2_fixture(output_dir)

            summary = annotate_v2_volume(
                ReadingV2Config(
                    output_dir=output_dir,
                    model="fake-model",
                    dry_run=True,
                    max_paragraphs_per_chunk=2,
                    max_dialogues_per_chunk=2,
                )
            )

            report = read_json(output_dir / "reading_v2" / "prompt_length_report.json")
            rows = list(
                read_jsonl(output_dir / "reading_v2" / "prompt_length_report.jsonl")
            )
            prompts = list((output_dir / "reading_v2" / "prompts").glob("*.txt"))
            self.assertEqual(summary["pipeline"], "reading-v2")
            self.assertEqual(summary["prompt_count"], 6)
            self.assertEqual(summary["prompt_over_limit_count"], 0)
            self.assertEqual(report["prompt_count"], 6)
            self.assertEqual(report["task_counts"]["annotation"], 1)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(prompts), 6)

    def test_annotate_v2_cache_only_writes_annotations_and_memory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output_dir = Path(tmp) / "outputs" / "volume_01"
            self._write_v2_fixture(output_dir)
            cache_dir = output_dir / "reading_v2" / "cache"
            request_prefix = "volume_01-r000001"
            model_suffix = "fake_model"
            write_json(
                cache_dir / f"{request_prefix}--annotation--{model_suffix}.json",
                {
                    "annotations": [
                        {
                            "dialogue_id": "volume_01-d000001",
                            "speaker_entity_id": "char_0001",
                            "speaker_display": "Alice",
                            "speaker_status": "known",
                            "confidence": 0.91,
                            "evidence": ["Narration attributes the line to Alice."],
                            "needs_review": False,
                        }
                    ]
                },
            )
            write_json(
                cache_dir / f"{request_prefix}--chunk_summary--{model_suffix}.json",
                {
                    "chunk_id": request_prefix,
                    "summary": "Alice greets someone.",
                    "active_entities": ["char_0001"],
                    "open_questions": [],
                    "evidence_refs": ["volume_01-d000001"],
                },
            )
            write_json(
                cache_dir / f"{request_prefix}--entity_discovery--{model_suffix}.json",
                {
                    "new_entities": [
                        {
                            "entity_id": "char_0001",
                            "entity_type": "character",
                            "display_name": "Alice",
                            "importance": "medium",
                            "summary": "A named speaker in the current chunk.",
                            "evidence_refs": ["volume_01-d000001"],
                            "dialogue_count_delta": 1,
                        }
                    ],
                    "merge_candidates": [],
                },
            )
            write_json(
                cache_dir / f"{request_prefix}--entity_update--{model_suffix}.json",
                {
                    "updates": [
                        {
                            "entity_id": "char_0001",
                            "summary": "Alice greets someone.",
                            "importance": "medium",
                            "dialogue_count_delta": 1,
                            "latest_seen_chunk_id": request_prefix,
                            "relationship_updates": [],
                            "evidence_refs": ["volume_01-d000001"],
                        }
                    ]
                },
            )
            write_json(
                cache_dir / f"{request_prefix}--global_summary--{model_suffix}.json",
                {
                    "summary": "Alice has appeared and greeted someone.",
                    "new_facts": ["Alice speaks in the opening chunk."],
                    "retained_facts": [],
                    "dropped_facts_reason": [],
                },
            )
            write_json(
                cache_dir / f"{request_prefix}--repair--{model_suffix}.json",
                {"repairs": []},
            )

            summary = annotate_v2_volume(
                ReadingV2Config(
                    output_dir=output_dir,
                    model="fake-model",
                    cache_only=True,
                    write_prompts=False,
                )
            )

            annotations = list(
                read_jsonl(output_dir / "annotation_v2" / "annotations.jsonl")
            )
            characters = list(read_jsonl(output_dir / "memory_v2" / "characters.jsonl"))
            labeled = (output_dir / "annotation_v2" / "final_labeled.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(summary["annotation_count"], 1)
            self.assertEqual(summary["failed_request_count"], 0)
            self.assertEqual(annotations[0]["speaker_display"], "Alice")
            self.assertFalse(annotations[0]["needs_review"])
            self.assertEqual(characters[0]["display_name"], "Alice")
            self.assertIn("【Alice】", labeled)


class CliOutputPathTests(unittest.TestCase):
    def test_output_root_selects_volume_directory(self) -> None:
        args = Namespace(output=None, output_root=Path("outputs2"), volume=1)

        self.assertEqual(_resolve_output(args), Path("outputs2") / "volume_01")

    def test_explicit_output_overrides_output_root(self) -> None:
        args = Namespace(
            output=Path("custom") / "book01",
            output_root=Path("outputs2"),
            volume=1,
        )

        self.assertEqual(_resolve_output(args), Path("custom") / "book01")


if __name__ == "__main__":
    unittest.main()
