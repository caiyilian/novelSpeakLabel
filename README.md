# novelSpeakLabel

轻小说对话说话人标注实验项目。

当前代码只实现 issue #1 要求的前两步：

- 阶段 0：预处理，把小说拆成章节、场景、段落和对话单元。
- 阶段 1：发现与建库，用 Ollama 扫描场景，生成角色候选、未知人物、场景摘要和记忆文件。

原文不会被原地修改，所有产物默认写到 `outputs/volume_XX/`。

## 运行第一卷

先做纯本地预处理：

```bash
python -m novel_speaker_label preprocess --volume 1
```

这会读取 `novels` 目录中包含 `01` 的 txt，并输出：

```text
outputs/volume_01/
  source.txt
  volume.json
  preprocess/
    chapters.jsonl
    paragraphs.jsonl
    scenes.jsonl
    dialogues.jsonl
```

再做阶段 1 角色发现：

```bash
python -m novel_speaker_label discover --volume 1 --model qwen3:32b
```

如果只想检查 prompt，不调用 Ollama：

```bash
python -m novel_speaker_label discover --volume 1 --model qwen3:32b --dry-run
```

阶段 1 会输出：

```text
outputs/volume_01/
  discovery/
    prompts/
    cache/
    raw_responses.jsonl
    scene_discoveries.jsonl
    run_summary.json
  memory/
    aliases.jsonl
    mystery_entities.jsonl
    episodic/
      scenes.jsonl
    semantic/
      characters.jsonl
```

也可以一步跑完：

```bash
python -m novel_speaker_label run-volume --volume 1 --model qwen3:32b
```

## Ollama 参数

默认 Ollama 地址是 `http://127.0.0.1:11434`，可以改：

```bash
python -m novel_speaker_label discover \
  --volume 1 \
  --ollama-host http://127.0.0.1:11434 \
  --model qwen3:32b \
  --timeout 120 \
  --temperature 0.0 \
  --num-predict 8192
```

阶段 1 会按场景写入 `discovery/cache/*.json`，重复运行时默认复用缓存。需要重跑模型时加：

```bash
python -m novel_speaker_label discover --volume 1 --overwrite-cache
```

## 本地测试

```bash
python -m unittest discover -s tests
```
