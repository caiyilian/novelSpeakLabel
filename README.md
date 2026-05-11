# novelSpeakLabel

轻小说对话说话人标注实验项目。

当前代码实现前三步中的自动化部分：

- 阶段 0：预处理，把小说拆成章节、场景、段落和对话单元。
- 阶段 1：发现与建库，用 Ollama 扫描场景，生成角色候选、未知人物、场景摘要和记忆文件。
- 阶段 2：正式标注，基于阶段 1 记忆逐句判断说话人，生成投票、聚合结果、复核队列和标注文本。

原文不会被原地修改，所有产物默认写到 `outputs/volume_XX/`。如果想保存到另一套目录，可以给所有阶段传 `--output-root outputs2`，输出会变成 `outputs2/volume_XX/`；也可以用 `--output outputs2/volume_01` 指定完整卷目录。cache 读取也跟随这个输出目录。

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

阶段 1 默认会把大场景继续切成较小请求，避免一次给本地模型塞入过长 prompt。

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
    raw/
    failures/
    failed_requests.jsonl
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

`memory/semantic/characters.jsonl` 会保存角色名、别名、身份、关系、说话风格和 `speech_markers`。`speech_markers` 用来记录高区分度的口癖、自称、固定句尾或固定短语，阶段 2 会把它作为候选说话人的辅助证据。如果你手里是旧版阶段 1 缓存，需要用 `discover --overwrite-cache` 重新跑一次，才会提取新的口癖字段。

然后做阶段 2 正式标注：

```bash
python -m novel_speaker_label annotate --volume 1 --model qwen3:32b
```

阶段 2 默认会把同一场景内相邻的多句台词放在一个窗口里一起判断，而不是逐句独立判断。这样模型能看到交易寒暄、问答、命令/回应等连续轮次，减少相邻说话人串错。默认窗口大小是 8 句，可以调整：

```bash
python -m novel_speaker_label annotate \
  --volume 1 \
  --model qwen3:32b \
  --annotation-window-size 8 \
  --context-paragraph-radius 3 \
  --scene-summary-radius 1
```

也可以启用新版结构化 Stage 2，让流程拆成“证据抽取 -> 裁判 -> 反证检查”三步。单模型显存有限时可以让同一个模型分三次扮演不同角色：

```bash
python -m novel_speaker_label annotate \
  --volume 1 \
  --pipeline structured \
  --model qwen3:32b
```

如果有多模型资源，可以显式分工：

```bash
python -m novel_speaker_label annotate \
  --volume 1 \
  --pipeline structured \
  --evidence-model qwen3:32b \
  --judge-model qwen3-coder:30b \
  --contradiction-model qwen2.5-coder:32b
```

结构化模式会额外写出 `annotation/evidence.jsonl`、`annotation/judgements.jsonl` 和 `annotation/contradiction_checks.jsonl`，对应的 cache 目录是 `evidence_cache/`、`judgement_cache/`、`contradiction_cache/`。

如果只想检查阶段 2 prompt，不调用 Ollama：

```bash
python -m novel_speaker_label annotate --volume 1 --model qwen3:32b --dry-run
```

阶段 2 会输出：

```text
outputs/volume_01/
  annotation/
    prompts/
    cache/
    raw/
    failures/
    evidence.jsonl
    judgements.jsonl
    contradiction_checks.jsonl
    votes.jsonl
    annotations.jsonl
    review_queue.jsonl
    failed_requests.jsonl
    run_summary.json
    final_labeled.txt
```

`final_labeled.txt` 是新文件，原文不会被原地修改。`review_queue.jsonl` 中的对话需要人工或后续阶段 3 回填修正。
单模型运行时默认要求 `confidence >= 0.75` 才直接接受，否则进入 `review_queue.jsonl`；想更激进或更保守可以调 `--min-confidence`。

也可以一步跑完：

```bash
python -m novel_speaker_label run-volume --volume 1 --model qwen3:32b
```

`run-volume` 目前只串起阶段 0 和阶段 1；阶段 2 请在阶段 1 完成后单独运行 `annotate`。

## Ollama 参数

默认 Ollama 地址是 `http://127.0.0.1:11434`，可以改：

```bash
python -m novel_speaker_label discover \
  --volume 1 \
  --ollama-host http://127.0.0.1:11434 \
  --model qwen3:32b \
  --timeout 1800 \
  --temperature 0.0 \
  --num-predict 4096
```

`--timeout` 是 HTTP socket 超时，不是整卷运行时间上限。本地大模型处理长 prompt 时可能很久才生成完，超时时可以：

```bash
python -m novel_speaker_label discover --volume 1 --model qwen3:32b --timeout 0
```

`--timeout 0` 表示禁用 socket 超时。

默认每次请求最多发送 30 个段落和 40 个对话单元，可以按服务器速度调小：

```bash
python -m novel_speaker_label discover \
  --volume 1 \
  --model qwen3:32b \
  --max-paragraphs-per-request 15 \
  --max-dialogues-per-request 20
```

阶段 1 会按场景写入 `discovery/cache/*.json`，重复运行时默认复用缓存。需要重跑模型时加：

```bash
python -m novel_speaker_label discover --volume 1 --overwrite-cache
```

如果某个请求超时或模型输出无法解析，默认会记录到 `discovery/failures/` 和 `discovery/failed_requests.jsonl`，然后继续后面的请求。想遇到第一个错误就停止，可以加：

```bash
python -m novel_speaker_label discover --volume 1 --stop-on-error
```

如果已经跑完 Ollama，只想用现有 `discovery/cache/*.json` 按当前代码重新生成角色库和记忆文件，不重写 prompts，也不再调用 Ollama：

```bash
python -m novel_speaker_label rebuild-memory \
  --volume 1 \
  --model qwen3:32b \
  --max-paragraphs-per-request 15 \
  --max-dialogues-per-request 20
```

注意：`--max-paragraphs-per-request` 和 `--max-dialogues-per-request` 必须和生成 cache 时使用的切分参数一致，否则会找不到对应的 cache 文件。

阶段 2 也支持只用现有 cache 重建聚合结果和标注文本：

```bash
python -m novel_speaker_label annotate --volume 1 --model qwen3:32b --cache-only
```

如果要读取 `outputs2` 里的 cache：

```bash
python -m novel_speaker_label annotate --volume 1 --output-root outputs2 --model qwen3:32b --cache-only
```

窗口级标注会生成新的 cache 文件名。若需要复用旧版逐句标注 cache，请显式使用单句窗口：

```bash
python -m novel_speaker_label annotate \
  --volume 1 \
  --model qwen3:32b \
  --cache-only \
  --annotation-window-size 1
```

多模型投票可以重复传 `--model`，并可按模型设置聚合权重：

```bash
python -m novel_speaker_label annotate \
  --volume 1 \
  --model qwen3:32b \
  --model llama3:latest \
  --model-weight qwen3:32b=1.0 \
  --model-weight llama3:latest=0.8
```

## 本地测试

```bash
python -m unittest discover -s tests
```
