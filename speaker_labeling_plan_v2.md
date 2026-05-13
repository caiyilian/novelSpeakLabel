# 方案 V2：实时阅读式说话人标注

本文是 issue #4 的设计方案，不是代码实现。目标是把“像人类一样顺序阅读小说、边读边形成记忆、边回看刚读过的对话”的思路整理成可落地的工程方案，后续再按本文拆分实现。

V2 不否定现有 `speaker_labeling_plan.md` 和阶段 2 structured pipeline。它是一条新的实验路线：用更接近人类阅读过程的顺序状态机做说话人标注，输出可人工扫读的结果和必要统计，先让人肉眼判断效果是否值得继续。

## 1. 目标

### 1.1 要解决的问题

当前阶段 2 的主要问题不是“模型完全不会判断”，而是判断方式和人类阅读小说的方式不一致：

- 旧方案倾向于先扫描/建库，再在局部窗口中裁判。
- 人类读小说是顺序读：一开始不知道角色，随着叙述推进逐步建立角色、场景和关系。
- 很多台词在刚出现时不能立刻精确命名，但可以先判断“不是某个重要角色”“像无名群体/低重要度群体”“像同一轮对话中的另一方”。
- 后文出现身份介绍后，人类会回头修正刚才的理解。

V2 的目标是把这个过程工程化。

### 1.2 非目标

- 不追求单次模型调用完成全卷标注。
- 不要求模型永远输出具体角色名；允许保留 `unknown`、`mystery`、`npc_group`。
- 不在 V2 初版里直接删除旧 pipeline。
- 不把摘要当成唯一记忆来源；摘要只是压缩视图，不能覆盖结构化事实。
- 不为了降低 `review_count` 牺牲重要角色的准确率。

## 2. 核心原则

### 2.1 Python 总控，LLM 只做子任务

V2 仍然由 Python orchestrator 控制流程：

- 分块阅读
- prompt 预算控制
- 缓存和重试
- JSON 校验
- 摘要长度检查
- 角色实体合并
- 对话标注聚合
- 输出最终文件

LLM 不做全局调度，只做受约束的单次任务，例如“给当前 chunk 标注说话人”“更新长期摘要”“判断新角色是否应建卡”。

### 2.2 顺序阅读，但不是纯串行迷信

V2 模拟顺序阅读，但不能让每一步的错误无限传播。设计上采用：

- 当前 chunk 原文始终作为最强证据。
- 长期摘要只提供背景，不允许推翻当前 chunk 明确叙述。
- 角色卡片必须带 evidence id，避免摘要漂移后无法追溯。
- 对话标注先发生，再用角色/摘要更新结果辅助回看，而不是让摘要先污染标注。

### 2.3 允许未知，但未知要可管理

V2 中的未知不是一个笼统标签，而分三类：

- `unknown`: 当前确实无法判断，也没有稳定身份线索。
- `mystery`: 可追踪的未知个体，后文可能合并到正式角色。
- `npc_group`: 无名群体、路人、临时职能角色、围观者、队伍成员等低重要度群体。

这样能避免把所有不确定台词混成一个“未知”，也避免为了普通路人过度建库。

### 2.4 通用小说优先，具体作品只作例子

V2 的最终目标是通用小说角色标注，不是某一部小说的专用标注器。因此后续 prompt 和规则必须遵守：

- 不硬编码特定作品角色、地点、职业、关系或剧情。
- 文档中的具体作品片段只作为解释样例，不能直接写进运行时 prompt。
- `npc_group` 的判断依据应是“无名、低重要度、群体或临时职能”，不是某个作品里的固定身份。
- 作品相关知识只能来自当前输入文本和运行过程中形成的记忆，不允许预置外部剧情知识。
- 如果某个规则看起来只对单本书有效，应降级为示例或测试用例，不能进入通用 prompt。

## 3. 阅读单位设计

### 3.1 chunk 是 V2 的基本处理单位

V2 不按单句处理，也不按整卷处理，而按 chunk 顺序阅读。chunk 应尽量保持叙事连续：

- 默认按场景切分。
- 场景太长时按段落数和对话数再切子 chunk。
- 场景太短时可以合并相邻小段，但不能跨越明显章节/场景边界。

建议初始参数：

- `max_paragraphs_per_chunk`: 12 到 20
- `max_dialogues_per_chunk`: 20 到 40
- `lookback_paragraphs`: 2 到 4
- `lookahead_paragraphs`: 0 到 2

V2 是顺序阅读，默认不使用大量 lookahead。少量 lookahead 只用于处理“说完后下一句叙述才说明是谁说”的情况。

### 3.2 chunk 输入包含三种记忆

每个 chunk 的瞬时输入：

- 当前 chunk 原文段落
- 当前 chunk 对话列表
- 少量前文段落
- 必要时少量后文段落

短期记忆：

- 上一个 chunk 的短期摘要
- 当前场景最近若干轮对话状态
- 最近出现角色列表

长期记忆：

- 卷级长期摘要
- 相关角色卡片
- 相关 mystery 卡片
- 相关 npc group 卡片

## 4. 每个 chunk 的任务链

issue #4 提到一个 chunk 可能调用五次模型。V2 建议保留“多子任务”方向，但调整顺序，降低串行误差传播。

### 4.1 Task A：当前 chunk 初标注

目标：给当前 chunk 内每条对话做初始说话人判断。

输入：

- 当前 chunk 原文和对话列表
- 上一 chunk 短期摘要
- 当前长期摘要的压缩版
- 筛选后的角色卡片
- 筛选后的 mystery / npc group 卡片
- 固定标注规则

输出：

```json
{
  "annotations": [
    {
      "dialogue_id": "volume_01-d000001",
      "speaker_entity_id": "char_known_001",
      "speaker_display": "已知角色A",
      "speaker_status": "known|mystery|npc_group|unknown|review",
      "confidence": 0.0,
      "evidence": ["短证据，不复制大段原文"],
      "negative_evidence": ["为什么不是某角色"],
      "is_backfillable": true,
      "needs_review": false
    }
  ],
  "chunk_notes": "本 chunk 对话轮次判断摘要"
}
```

关键规则：

- 被称呼者通常不是说话者。
- “某角色名 + 尊称 + 你”这类直接称呼表达，应作为“不是该角色本人”的强反证。
- 如果只能判断“无名群体/低重要度临时角色”，应输出 `npc_group`，不要硬建 mystery。
- 对重要角色宁可 `review`，也不要误标成 npc。
- 对普通路人宁可 `npc_group`，不要过度追求精确身份。

### 4.2 Task B：当前 chunk 短期摘要

目标：总结刚读完的 chunk，作为下一 chunk 的短期记忆。

输入：

- 当前 chunk 原文
- Task A 的标注结果
- 当前 chunk 中新出现/活跃的角色简表

输出：

```json
{
  "chunk_id": "volume_01-r0001",
  "summary": "本 chunk 发生的事件和对话关系",
  "active_entities": ["char_known_001", "npc_group_local_people"],
  "open_questions": ["第一句交易发问者仍不确定"],
  "evidence_refs": ["paragraph_id", "dialogue_id"]
}
```

短期摘要可以较具体，因为生命周期短。它服务于后续几个 chunk，而不是永久事实库。

### 4.3 Task C：长期摘要更新

目标：用旧长期摘要 + 当前 chunk 短期摘要，更新卷级长期摘要。

输入：

- 旧长期摘要
- 当前 chunk 短期摘要
- 当前 chunk 的关键结构化事实

输出：

```json
{
  "summary": "更新后的长期摘要",
  "new_facts": ["新增事实"],
  "retained_facts": ["仍重要的旧事实"],
  "dropped_facts_reason": ["哪些细节被压缩掉以及原因"]
}
```

防漂移规则：

- 长期摘要不能凭空新增当前 chunk 没有的事实。
- 如果新摘要与结构化事实冲突，Python 应拒绝并要求模型修正。
- 长期摘要不保存所有人物细节；人物细节进入角色卡片。
- 长期摘要应保存故事进展、当前地点、主要关系和未解决事件。

### 4.4 Task D：新增实体发现

目标：从当前 chunk 中发现新角色、可追踪未知人物和普通群体 NPC。

输入：

- 当前 chunk 原文
- Task A 标注
- 当前已有角色/未知实体简表

输出：

```json
{
  "new_entities": [
    {
      "entity_type": "character|mystery|npc_group",
      "display_name": "已知角色A",
      "aliases": [],
      "importance": "minor|medium|major",
      "summary": "已知角色A在本段中被介绍出身份或行动目标",
      "evidence_refs": ["paragraph_id"],
      "dialogue_count_delta": 2
    }
  ],
  "merge_candidates": [
    {
      "source_entity_id": "mystery_001",
      "target_entity_id": "char_lawrence",
      "reason": "后文身份介绍指向同一人",
      "confidence": 0.0
    }
  ]
}
```

实体发现规则：

- 有名字的人物才优先建 `character`。
- 没名字但连续说话、后文可能揭示身份的个体建 `mystery`。
- 无名群体、路人、临时职能角色、围观者、队伍成员等默认建 `npc_group`。
- 不要把普通职业/称谓都建成角色，例如“交易者”“行人”“年轻人”需先看是否可追踪。

### 4.5 Task E：已有角色卡片更新

目标：更新本 chunk 出现过的旧角色卡片，包括摘要、对话数、最近出现位置和重要性。

输入：

- 当前 chunk 短期摘要
- 长期摘要
- 当前 chunk 标注结果
- 本 chunk 出现过的角色旧卡片

输出：

```json
{
  "updates": [
    {
      "entity_id": "char_known_001",
      "summary": "已知角色A在当前事件中继续行动，并与其他角色产生对话关系",
      "importance": "major",
      "dialogue_count_delta": 2,
      "latest_seen_chunk_id": "volume_01-r0001",
      "relationship_updates": [],
      "evidence_refs": ["paragraph_id"]
    }
  ]
}
```

重要性规则：

- 只允许 `minor`、`medium`、`major` 三档。
- 有名字角色默认至少 `medium` 候选，但不能自动 `major`。
- `major` 需要持续叙事中心、频繁对话、长期关系线或标题级重要性。
- `minor` 到 `medium`、`medium` 到 `major` 可以升级。
- 降级要有滞后机制，避免相邻 chunk 抖动。

建议使用稳定规则：

- 连续 3 个以上 chunk 未出现，才允许 `major -> medium`。
- 连续 6 个以上 chunk 未出现，才允许 `medium -> minor`。
- 主角、长期同行者、已确认核心角色不自动降级。

### 4.6 Task F：当前 chunk 回看补标

目标：在 Task B/C/D/E 完成后，回看当前 chunk 中仍为 `unknown`、`mystery`、`review` 的对话，尝试补标。

输入：

- 当前 chunk 原文
- Task A 初标注
- 更新后的短期摘要
- 更新后的长期摘要
- 更新后的角色/未知实体卡片
- 只包含待处理对话的列表

输出：

```json
{
  "repairs": [
    {
      "dialogue_id": "volume_01-d000001",
      "previous_speaker_entity_id": "unknown",
      "new_speaker_entity_id": "npc_group_local_people",
      "speaker_status": "npc_group",
      "confidence": 0.82,
      "reason": "当前场景中与已知角色轮流说话的无名低重要度群体",
      "stop_reason": "resolved"
    }
  ]
}
```

停止规则：

- 如果仍无法判断，保留 `unknown` 或 `review`。
- 如果只知道是普通群体，允许标成 `npc_group`。
- 如果补标会把重要角色误判为路人，保留 `review`。
- 不跨未来 chunk 大范围回填；当前任务只回看刚读完的 chunk。

## 5. 推荐的执行顺序

每个 chunk 的推荐顺序：

1. 选取 chunk 和 prompt 候选记忆。
2. Task A：初标注。
3. Task B：生成 chunk 短期摘要。
4. Task D：新增实体发现。
5. Task E：更新已有角色卡片。
6. Task C：更新长期摘要。
7. Task F：回看补标当前 chunk。
8. Python 写入 annotations、memory、review_queue。

说明：

- Task C 放在 Task D/E 后，是为了让长期摘要参考已经结构化过的人物变化。
- Task F 放最后，是为了模拟“读完这一小段后再回头看刚才不确定台词”。
- 如果后续实测发现长期摘要应更早更新，可以调整，但初版先避免摘要先污染实体判断。

## 6. Prompt 预算和上下文控制

### 6.1 不只看字符数

issue #4 提到 qwen3:32b context length 是 40960。V2 不能只用中文字数判断 prompt 是否安全，因为 JSON、标点、英文 key、系统提示都会占上下文。

初版建议采用两层预算：

- 粗略字符预算：快速过滤。
- token 估算预算：如果本地没有 tokenizer，先用保守估算。

保守估算规则：

```text
estimated_tokens = ceil(chinese_chars * 1.6 + ascii_chars * 0.4 + json_punctuation * 0.6)
```

这不是精确 tokenizer，但足够避免贴近上限。

### 6.2 安全上限

对 qwen3:32b，初版不要接近 40960。建议：

- 单次 prompt 目标上限：24000 estimated tokens
- 硬上限：30000 estimated tokens
- 超过硬上限时，Python 必须缩减候选角色、减少上下文段落或分裂 chunk。

### 6.3 角色候选筛选

Task A 的 prompt 最容易变长。候选角色筛选建议：

优先级从高到低：

1. 当前 chunk 明确提到的角色。
2. 上一 chunk 出现的角色。
3. 当前场景活跃角色。
4. `major` 角色。
5. 最近 N 个 chunk 出现过的 `medium` 角色。
6. 与当前地点/事件相关的 mystery。

不要把所有角色卡片塞进 prompt。对于长期未出现的 `minor` 角色，只给极短索引，允许模型输出“可能是未列出的旧角色”，再由 Python 二次检索。

### 6.4 代码完成标准：必须生成 prompt 长度报告

后续实现 V2 时，不能只写完代码就算完成。至少需要跑一次 dry-run，让 Python 生成实际 prompt，并输出长度检查报告。

建议 dry-run 产物：

```text
reading_v2/
  prompt_length_report.json
  prompt_length_report.jsonl
```

报告字段建议包含：

```json
{
  "chunk_id": "volume_01-r0001",
  "task": "annotation|chunk_summary|entity_discovery|entity_update|global_summary|repair",
  "prompt_path": "reading_v2/prompts/...",
  "char_count": 0,
  "estimated_tokens": 0,
  "target_token_limit": 24000,
  "hard_token_limit": 30000,
  "status": "ok|near_limit|over_limit"
}
```

完成标准：

- 所有 prompt 必须生成成功。
- 所有 prompt 的 `status` 必须是 `ok` 或最多少量 `near_limit`，不能有 `over_limit`。
- 如果出现 `over_limit`，这次代码实现不能算完成，必须先修 prompt/chunk 切分。
- `run_summary.json` 中要记录最大 prompt 长度、平均 prompt 长度、超限数量和 near-limit 数量。

### 6.5 prompt 过长时先排查 preprocess/chunk 切分

prompt 过长不能只靠删 prompt 文案硬压。排查顺序应是：

1. 检查 preprocess 产物中是否有异常长段落。
2. 检查章节/场景切分是否把多个自然场景粘在一起。
3. 检查 chunk 是否包含过多段落或过多对话。
4. 检查 lookback/lookahead 是否过大。
5. 检查候选角色筛选是否放入了过多不相关角色卡。
6. 最后才考虑压缩 prompt 模板文案。

如果发现 preprocess 的段落或场景切分本身不合理，应优先修切分或在 V2 chunk 层做二次拆分，而不是把超长输入直接交给模型。

## 7. 摘要长度控制

### 7.1 不只靠 prompt 要求长度

模型不可靠地感知“500 到 600 字”。Python 应计算输出长度，并触发压缩或扩写。

初版建议：

- chunk 短期摘要：250 到 450 中文字符。
- 长期摘要：450 到 700 中文字符。
- 角色摘要：80 到 180 中文字符。
- npc group 摘要：40 到 120 中文字符。

这些数字不是最终指标，后续可以根据效果调。

### 7.2 压缩循环

如果摘要过长：

1. 保留原摘要。
2. 要求模型在不丢失关键事实的前提下压缩。
3. Python 重新计数。
4. 最多压缩 2 次。
5. 仍过长时，用 Python 截断低优先级字段，而不是无限循环。

如果摘要过短：

1. 要求模型补充地点、人物、事件、未解决问题。
2. 最多扩写 1 次。
3. 仍过短也可以接受，不能为了长度编造事实。

### 7.3 防止压缩丢事实

摘要之外必须保留结构化事实表：

- `facts.jsonl`
- `entity_events.jsonl`
- `dialogue_links.jsonl`

摘要用于 prompt，人类可读；结构化事实用于追溯和防漂移。

## 8. 记忆结构

V2 建议新增独立目录，不污染现有 `memory/`：

```text
outputs2/volume_01/
  reading_v2/
    chunks.jsonl
    run_summary.json
    prompts/
    raw/
    cache/
  memory_v2/
    global_summary.json
    chunk_summaries.jsonl
    facts.jsonl
    characters.jsonl
    mysteries.jsonl
    npc_groups.jsonl
    entity_events.jsonl
  annotation_v2/
    annotations.jsonl
    repairs.jsonl
    review_queue.jsonl
    final_labeled.txt
    run_summary.json
```

这样 V2 可以和现有方案并行运行，不会互相覆盖 cache，也方便单独拿 V2 输出给人工扫读。

## 9. 角色库防脏增长

### 9.1 实体类型必须分清

角色库拆成三张表：

- `characters.jsonl`: 有名字或稳定身份的重要/中等人物。
- `mysteries.jsonl`: 可追踪但暂未命名的个体。
- `npc_groups.jsonl`: 普通群体或不重要路人。

不要把这三类混在同一个列表里让模型自由选择。

### 9.2 建卡门槛

新建 `character`：

- 有明确名字，或
- 后文明确身份，或
- 多次出现并影响剧情。

新建 `mystery`：

- 同一未知个体在多个对话或段落中可追踪，且可能后续揭示身份。

新建 `npc_group`：

- 群体说话人。
- 无名群体、路人、临时职能角色、围观者、队伍成员。
- 不需要追踪单个姓名。

### 9.3 合并流程

实体合并不让 Task D 直接改库。Task D 只输出 merge candidate，Python 再执行：

1. 检查 source/target 类型是否允许合并。
2. 检查 evidence refs 是否支持。
3. 检查是否会把重要角色合并进 npc_group。
4. 高置信自动合并，低置信进入 `entity_merge_review.jsonl`。

## 10. 对话标注策略

### 10.1 先接受“部分正确”

V2 不要求每句都有具体人名：

- 能确定是某个已知角色，就标该角色。
- 能确定不是某个重要角色但只是无名低重要度群体，就标对应 `npc_group`。
- 能确定是同一个未知个体，就标 `mystery_001`。
- 完全不确定，标 `unknown`。
- 可能误伤重要角色，标 `review`。

### 10.2 标注 confidence 只辅助，不做唯一门槛

模型 confidence 不能单独决定是否接受。Python 应结合：

- 是否有正证据。
- 是否有强反证。
- 是否涉及重要角色。
- 是否只是 npc_group。
- 是否和当前对话轮次一致。
- 是否后续 Task F 修正。

### 10.3 高风险误标

以下情况默认进入 review：

- 把已知重要角色的台词标成 npc_group，但证据不强。
- 被称呼者和说话者冲突没有解释。
- 当前 chunk 中同一轮对话出现不合理的连续发言。
- inline 引号无法确认是否独立对话。
- 拟声、仪式呼喊、群体喊声可能既不是具体角色也不是普通对话。

## 11. 和旧方案的关系

V2 与现有 structured pipeline 的关系：

- 旧 pipeline 保留，不在本阶段删除。
- V2 新增独立 pipeline，例如 `--pipeline reading-v2`。
- V2 可以复用 preprocess 的章节、段落、对话抽取结果。
- V2 不依赖旧 discovery 阶段先全卷建库。
- V2 可以读取旧 memory 作为可选 seed，但初版建议默认不读，避免旧错误污染对照实验。
- 本阶段不要求做自动化新旧方案对比；V2 只需要产出便于人工扫读的标注文本、复核队列和必要统计。

## 12. 实现阶段建议

### 阶段 A：文档和数据结构

- 新增本文档。
- 确定 V2 输出目录和 JSON schema。
- 确定 chunk 切分规则。

### 阶段 B：dry-run prompt 生成

- 新增 reading-v2 pipeline 骨架。
- 只生成 prompts，不调用 Ollama。
- 输出每个 task 的 prompt 长度估算。
- 验证不会超上下文预算。
- 如果 prompt 超限，必须先定位是 preprocess 切分问题、chunk 过大、候选角色过多，还是 prompt 模板本身过长。

### 阶段 C：单 chunk 端到端

- 只跑目标卷前 1 到 3 个 chunk。
- 实现 Task A 到 Task F。
- 写出 annotation_v2 和 memory_v2。
- 人工检查用户指定的代表性开头片段。

### 阶段 D：整卷运行

- 支持 cache-only。
- 支持失败续跑。
- 支持输出 final_labeled.txt。
- 记录每个 task 的调用次数、失败数、平均 prompt 长度、平均输出长度。

### 阶段 E：人工评估准备

输出给人工扫读的指标：

- 明显误标数量。
- 重要角色误标数量。
- `unknown` / `review` 是否合理。
- npc_group 是否减少无意义复核。
- 同一 mystery 是否被稳定追踪。
- 人工扫读体验是否更接近人类理解。
- prompt 长度报告是否全部在安全范围内。

## 13. 初版命令设计

建议新增命令：

```bash
python -m novel_speaker_label annotate-v2 --volume 1 --model qwen3:32b --output-root outputs_v2
```

可选参数：

```bash
--max-paragraphs-per-chunk 16
--max-dialogues-per-chunk 32
--lookback-paragraphs 3
--lookahead-paragraphs 1
--max-prompt-tokens 24000
--hard-prompt-tokens 30000
--cache-only
--dry-run
--overwrite-cache
--start-chunk-index 0
--chunk-limit 3
```

也可以复用现有 `annotate`：

```bash
python -m novel_speaker_label annotate --volume 1 --pipeline reading-v2 --model qwen3:32b
```

初版建议单独 `annotate-v2`，避免和现有 annotate 参数混在一起。

## 14. 关键风险和对应控制

### 14.1 摘要漂移

控制：

- 摘要必须引用 evidence refs。
- 结构化 facts 独立保存。
- 当前 chunk 明确原文优先于长期摘要。

### 14.2 串行误差传播

控制：

- Task A 初标注不依赖 Task B/C 的新输出。
- Task F 只修正当前 chunk 的不确定项。
- 重要角色高风险变更进 review。

### 14.3 角色库脏增长

控制：

- `character`、`mystery`、`npc_group` 分表。
- 建卡和合并由 Python 校验。
- 模型只提出候选，不直接改最终库。

### 14.4 重要性抖动

控制：

- 三档重要性。
- 升级可以快，降级必须慢。
- 主角/核心同行者锁定为 major。

### 14.5 prompt 过长

控制：

- 每次调用前估算 token。
- 角色候选按重要性和最近出现筛选。
- 超预算时缩短角色卡、缩小 chunk 或分裂任务。

### 14.6 过度标注普通路人

控制：

- 普通群体默认 `npc_group`。
- npc_group 不参与复杂身份合并，除非后文明确个体化。
- 输出时可显示为 `路人`、`非重要角色`、`无名群体`，或从当前文本中抽取出的通用群体称呼。

## 15. 建议的第一轮实验范围

第一轮不要直接跑完整卷。建议：

1. 用户指定的开头样例片段，优先包含普通对话、身份后置介绍、无名群体。
2. 主角或主要视角角色的身份介绍片段。
3. 重要角色首次登场和身份确认片段。

这些片段能覆盖：

- 无名群体 / 路人 / 临时职能角色
- 已知角色身份后置介绍
- 重要角色首次登场
- mystery 到 known 的可能回填
- 口癖/自称是否真的有价值

如果这些片段的人工扫读效果可以接受，再跑更大范围。

## 16. 最终判断标准

V2 是否值得继续，不看单个数字，而看人工扫读质量：

- 重要角色明显误标是否减少。
- 普通路人是否不再消耗大量复核。
- `unknown` 是否保留得合理。
- 后文身份揭示后，前文是否能稳定回填。
- 长期摘要是否保持事实稳定，没有自我扩写剧情。

如果 V2 在这些方面表现可接受，就继续实现更完整的回填修正。若 V2 更慢但更准，可以接受；若更慢且误标更多，则回到旧 structured pipeline 继续优化。
