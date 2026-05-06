from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from novel_speaker_label.jsonl import read_jsonl
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


if __name__ == "__main__":
    unittest.main()
