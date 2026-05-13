# V2 增强方案：多轮纠错与流程复核

本文回应 issue #5，目标是在不为具体作品写硬编码规则的前提下，继续增强 `speaker_labeling_plan_v2.md` 和当前 `annotate-v2` 流程的准确率。

结论先行：当前 V2 方向是对的，但还缺少一类关键任务：对已经给出高置信 `known` 的结果进行全局轮次复核。现有 repair 更像“补未知”，不是“查错”。issue #5 中的 bad case 大多不是模型不知道角色，而是模型把局部叙述反证、对话轮次和多人场景结构忽略了。

## 1. 当前流程核对

根据当前实现，`annotate-v2` 对每个有对话的 chunk 主要执行这些模型任务：

1. `annotation`：当前 chunk 初标注。
2. `chunk_summary`：当前 chunk 短期摘要。
3. `entity_discovery`：发现新增 character、mystery、npc_group。
4. `entity_update`：更新已有实体卡片。
5. `global_summary`：更新卷级长期摘要。
6. `mystery_resolution`：有 pending mystery 且已有 character 时，尝试合并。
7. `repair`：只在存在 `unknown`、`review`、`mystery` 或 `needs_review` 标注时回看补标。

因此实际已经不是简单的 Task A 到 Task E 五次调用，而是“五个基础任务 + 可选身份回填 + 条件 repair”。问题在于 repair 的触发条件太窄：如果 Task A 把整段多人对话高置信地标成同一个已知角色，后续不会进入 repair。

## 2. Bad Case 抽象

下面只抽象问题类型，不把具体小说角色写入运行时规则。

### 2.1 交易开场的问答反向

现象：前几句是一组交易问答，模型能识别双方存在，但可能把第一句或第二句的说话者反过来。

通用原因：

- chunk 开头缺少前置场景锚点。
- 第一轮对话没有显式 speaker tag。
- 模型用“主角视角”替代了“对话行为分析”。
- repair 只看到局部句子，没有强制重建整轮交易的问答结构。

应对方向：在复核任务里要求模型先重建 exchange：谁发起询问、谁回答数量/货物、谁表达感谢，再标注每句。

### 2.2 被询问者误判成提问者

现象：后文叙述明确说明某人“因为被询问而不悦”，但模型仍把前一句提问标给该人。

通用原因：

- 模型看到了场景中的职能名或称谓，就把该称谓当成 speaker。
- 没有把“被询问”“被问到”“听到这句话后”等叙述动作作为反证。

应对方向：给每条 dialogue 建立 `post_narration_evidence`，专门抽取下一到两段中“谁回答、谁皱眉、谁被询问、谁点头”等叙述反证。

### 2.3 打招呼动作归因反向

现象：叙述写明 A 朝 B 打招呼，模型却把引号内招呼语标成 B。

通用原因：

- 模型把“动作对象”误当成“动作发出者”。
- prompt 只有“被称呼者通常不是说话者”，但缺少“动作发出者通常是说话者”的正向建模。

应对方向：增加 narration-action 解析：`actor`、`action`、`target`、`linked_dialogue_id`。这不是硬编码某句话，而是通用叙述语义。

### 2.4 反问句归因反向

现象：后文叙述明确说明“女孩反问某人”，但模型仍把反问句标给被反问者。

通用原因：

- 模型没有把“反问 X”解析成“说话者不是 X”。
- 候选人之间缺少负证据比较。

应对方向：每条标注都必须输出 `negative_evidence`，并在复核任务中检查：如果一句话的证据显示“某人被问/被反问/被称呼”，却标成该人，则必须降为 review 或修正。

### 2.5 多人对话被压成同一人

现象：场景中有一对夫妻、主角加入谈话，模型却将连续多句都标成同一人，并给出高置信。

通用原因：

- Task A 允许模型直接输出每句 speaker，没有强制先建本 chunk 的 participant graph。
- 现有 repair 不检查高置信 `known`。
- `chunk_notes` 即使出现“所有对话均来自某人”这种明显可疑总结，也不会触发二次审查。

应对方向：新增整段对话轮次复核任务，重点处理“多参与者场景 + 连续同一 speaker + 叙述中提到其他谈话者”的组合风险。

### 2.6 职能称呼和实体合并

现象：同一局部人物可能被称为“某地点/组织的职能者”和简短职能名，模型会把它们当成两个实体。

通用原因：

- `entity_discovery` 更关注新增实体，较少做局部别名归并。
- 当前实体卡片缺少“称谓作用域”：有些称谓只在当前 scene 内可合并，不能全卷全局合并。

应对方向：新增 `alias_scope`：`global`、`scene_local`、`chunk_local`。职能称呼优先作为局部 alias 候选，而不是立刻建新角色。

## 3. 增强后的任务链

建议把当前流程升级为：

1. Task A：`annotation_draft`，初标注，但要求输出候选人与证据表。
2. Task B：`turn_structure`，重建当前 chunk 的对话轮次和参与者图。
3. Task C：`narration_evidence`，抽取每条 dialogue 前后叙述中的 speaker 正证据与反证据。
4. Task D：`speaker_adjudication`，结合 A/B/C 给出最终标注。
5. Task E：`chunk_summary`，生成短期摘要。
6. Task F：`entity_discovery`，发现新实体和 alias/merge 候选。
7. Task G：`entity_update`，更新角色卡。
8. Task H：`global_summary`，更新长期摘要。
9. Task I：`mystery_resolution`，处理 pending mystery。
10. Task J：`chunk_consistency_review`，无条件复核本 chunk 的全部标注。
11. Task K：`repair_apply`，只对 J 标出的 suspicious/dialogue 执行修正或降级。

如果担心调用次数过多，可以先合并 B/C 为一个 `evidence_extraction` 任务，保留 J 作为新增核心任务。准确率优先时，建议不要省掉 J。

## 4. 新增任务设计

### 4.1 Task B：对话轮次结构

目标：先理解这一段有几方在对话，而不是直接判定每句 speaker。

输入：

- 当前 chunk 原文。
- 当前 chunk dialogue 列表。
- 少量 lookback/lookahead。
- 现有候选实体卡片。

输出：

```json
{
  "participants": [
    {
      "participant_id": "p1",
      "entity_id": "char_or_mystery_or_npc_group_or_unknown",
      "display": "候选显示名",
      "role_in_scene": "questioner|answerer|listener|bystander|narrated_subject|unknown",
      "evidence_refs": ["paragraph_id"]
    }
  ],
  "turns": [
    {
      "dialogue_id": "volume_01-d000001",
      "likely_participant_ids": ["p1"],
      "turn_relation": "asks|answers|continues|interrupts|greets|reacts|monologue|unknown",
      "addressed_participant_ids": ["p2"],
      "confidence": 0.0,
      "reason": "短理由"
    }
  ],
  "scene_dialogue_pattern": "two_party|multi_party|group_response|monologue|unclear"
}
```

关键要求：

- 先判断是不是多人对话，再判断具体 speaker。
- 连续多句同一人不是禁止项，但需要能解释为什么不是问答轮替。
- 如果叙述出现“加入他们的对话”“夫妻”“众人”“另一方”等信号，必须把场景标成 `multi_party` 或 `group_response` 候选。

### 4.2 Task C：叙述证据抽取

目标：把模型容易忽略的叙述线索结构化。

输出：

```json
{
  "dialogue_evidence": [
    {
      "dialogue_id": "volume_01-d000001",
      "pre_narration": [
        {
          "type": "speaker_action|listener_action|address|scene_setup|turn_setup",
          "actor": "entity_or_participant_or_unknown",
          "target": "entity_or_participant_or_unknown",
          "polarity": "supports|contradicts|neutral",
          "evidence_ref": "paragraph_id",
          "note": "短证据"
        }
      ],
      "post_narration": [
        {
          "type": "answered_by|asked_person_reacts|speaker_said_done|listener_reacts",
          "actor": "entity_or_participant_or_unknown",
          "target": "entity_or_participant_or_unknown",
          "polarity": "supports|contradicts|neutral",
          "evidence_ref": "paragraph_id",
          "note": "短证据"
        }
      ]
    }
  ]
}
```

通用证据类型：

- `speaker_action`：某人说、问、回答、喊、打招呼、低语。
- `listener_action`：某人听到后、被问后、被称呼后、露出反应。
- `address`：台词中直接称呼某人。
- `turn_setup`：叙述说明某人加入对话、旁听、等待机会。
- `answered_by`：后文说明某人回答、点头、解释、拒绝。

这些是小说叙述的通用语义，不是某一本书的规则。

### 4.3 Task D：候选裁决

目标：把初标注从“直接给答案”改为“候选竞争 + 正反证据裁决”。

输出：

```json
{
  "annotations": [
    {
      "dialogue_id": "volume_01-d000001",
      "speaker_entity_id": "entity_id",
      "speaker_display": "显示名",
      "speaker_status": "known|mystery|npc_group|unknown|review",
      "confidence": 0.0,
      "candidate_scores": [
        {
          "entity_id": "entity_id",
          "display": "显示名",
          "supporting_evidence": ["短证据"],
          "contradicting_evidence": ["短反证"],
          "score": 0.0
        }
      ],
      "evidence": ["短证据"],
      "negative_evidence": ["为什么不是其他强候选"],
      "needs_review": false,
      "review_reason": ""
    }
  ]
}
```

裁决规则：

- 若 top1 和 top2 分差小，输出 `review`，不要强猜。
- 若当前 speaker 与叙述反证冲突，即使模型自信，也必须降置信或标 review。
- 若台词直接称呼某角色，被称呼者应作为强反候选，除非有明确自言自语或转述证据。
- 对 `npc_group` 的使用依然只基于“无名、低重要度、群体/职能、局部场景”，不得绑定具体作品。

### 4.4 Task J：全 chunk 一致性复核

这是本次增强的核心。

目标：不只修 unknown，也检查高置信 known 是否自洽。

输入：

- 当前 chunk 原文。
- Task D 最终标注。
- Task B 轮次结构。
- Task C 叙述证据。
- 当前实体卡片。
- 短期摘要和长期摘要。

输出：

```json
{
  "chunk_risk": "low|medium|high",
  "issues": [
    {
      "dialogue_id": "volume_01-d000001",
      "issue_type": "turn_inconsistency|narration_contradiction|address_conflict|unlikely_same_speaker_run|alias_conflict|low_evidence_known",
      "current_speaker_entity_id": "entity_id",
      "suggested_action": "keep|repair|downgrade_to_review|downgrade_to_unknown",
      "suggested_speaker_entity_id": "entity_id_or_empty",
      "confidence": 0.0,
      "reason": "短理由",
      "evidence_refs": ["paragraph_id", "dialogue_id"]
    }
  ],
  "safe_to_accept_without_repair": false
}
```

触发重点：

- `multi_party` 场景里出现长串同一 speaker，且中间有明显问答语义。
- 标注说话者同时也是台词中的被称呼者。
- 后文叙述显示某人是被问者、听者、反应者，却被标为上一句 speaker。
- evidence 只写“符合上下文轮次”，但没有具体 paragraph/dialogue 支持。
- `chunk_notes` 声称“所有对话均来自某人”，而原文出现其他明确谈话参与者。

注意：这些是风险触发器，不是直接改答案的 Python 硬规则。Python 只负责把风险送入模型复核；最终修正仍由模型基于文本证据输出。

### 4.5 Task K：定点修正

Task K 只处理 Task J 标出的 issue，不再只处理 unresolved。

输入：

- suspicious dialogues。
- 原标注。
- Task J reason。
- 当前 chunk 局部原文。
- 相关候选实体。

输出沿用现有 `repairs`，但增加 `repair_source`：

```json
{
  "repairs": [
    {
      "dialogue_id": "volume_01-d000001",
      "previous_speaker_entity_id": "old_id",
      "new_speaker_entity_id": "new_id",
      "speaker_display": "显示名",
      "speaker_status": "known|mystery|npc_group|unknown|review",
      "confidence": 0.0,
      "reason": "短理由",
      "stop_reason": "resolved|unchanged|needs_review",
      "repair_source": "chunk_consistency_review"
    }
  ]
}
```

## 5. 实体合并和别名策略

### 5.1 称谓不等于角色

对“职能称呼”“地点 + 职能”“组织 + 职能”等表达，先作为 alias 候选，而不是直接建全局角色。

建议实体卡增加：

```json
{
  "aliases": [
    {
      "name": "职能称呼",
      "scope": "global|scene_local|chunk_local",
      "evidence_refs": ["paragraph_id"],
      "confidence": 0.0
    }
  ]
}
```

### 5.2 合并任务要区分三类合并

1. `same_character`：同一个具名角色或已确认个体。
2. `same_scene_role`：同一局部场景里的同一职能者，不保证跨场景。
3. `same_group_type`：同类 NPC group，不代表同一批人。

只有 `same_character` 可以重写历史标注为同一 character。`same_scene_role` 只在当前 scene 内合并显示名。`same_group_type` 只用于统计和输出整洁，不应制造单一角色。

### 5.3 合并输出

```json
{
  "alias_candidates": [
    {
      "source_name": "短称谓",
      "target_entity_id": "entity_id",
      "merge_type": "same_character|same_scene_role|same_group_type",
      "scope": "global|scene_local|chunk_local",
      "confidence": 0.0,
      "reason": "短证据",
      "evidence_refs": ["paragraph_id"]
    }
  ]
}
```

## 6. Python 编排改动建议

### 6.1 不再只按 status 触发 repair

当前逻辑只把 `unknown`、`review`、`mystery` 和 `needs_review` 送入 repair。建议改为：

1. 始终运行 `chunk_consistency_review`，或至少在风险条件命中时运行。
2. 把 review 输出的 `suggested_action != keep` 加入 repair 输入。
3. `repair` 的输入字段从 `unresolved_annotations` 泛化为 `repair_targets`。

示意：

```python
review_targets = [
    issue for issue in chunk_review["issues"]
    if issue["suggested_action"] in {"repair", "downgrade_to_review", "downgrade_to_unknown"}
]
repair_targets = [
    *unresolved_annotations,
    *review_targets,
]
```

### 6.2 增加风险筛选，但不写死答案

为了控制调用成本，可以由 Python 做轻量风险筛选，只决定“是否需要复核”，不决定“speaker 是谁”。

可用风险特征：

- 当前 chunk 对话数大于等于 3。
- 当前 chunk 初标注 speaker 唯一，但原文出现两个以上候选参与者。
- `chunk_notes` 或摘要包含“所有对话”这类绝对判断。
- 多个 annotation 的 evidence 完全相同或高度空泛。
- known 高置信但 `negative_evidence` 为空。
- dialogue 文本中直接称呼当前 speaker 的 display 或 alias。

这些风险特征不包含具体作品、具体人物，也不直接修正输出，因此不属于针对性硬编码。

### 6.3 输出更多可审计产物

建议新增：

```text
reading_v2/
  turn_structure/
  narration_evidence/
  chunk_review/
annotation_v2/
  consistency_issues.jsonl
  repair_targets.jsonl
```

这样人工能看到模型为什么复核、修了什么、哪些仍保留 review。

## 7. Prompt 修改方向

### 7.1 初标注 prompt

初标注不应只要求“给 speaker”，还要要求：

- 每句至少列出两个候选，除非只有一个合理候选。
- 每句必须写 `negative_evidence`。
- 对多人场景必须先说明参与者。
- 对连续同一 speaker 必须说明为什么不是轮替。

### 7.2 复核 prompt

复核 prompt 应避免让模型重复 Task A 的思路，而是明确扮演审稿人：

- 找矛盾，不是重新完整标注。
- 专查高置信但证据弱的 known。
- 专查叙述反证、被称呼者、问答轮次、多人场景。
- 允许输出“保持原标注”，但必须给出证据。

### 7.3 修正 prompt

修正 prompt 应要求：

- 只修改 `repair_targets`。
- 不因为目标被送修就强行改。
- 证据不足时降级到 `review`，不要猜。
- 如果建议改成局部职能者，优先使用 `mystery` 或 `npc_group`，并带上局部 scope。

## 8. 评估方案

先不要追求整卷完全自动评估。建议建立小型 bad case set：

```text
eval_cases/
  issue_5_bad_cases.jsonl
```

每条包含：

```json
{
  "case_id": "issue5-001",
  "chunk_id": "volume_01-r000040",
  "dialogue_ids": ["volume_01-d000228", "volume_01-d000229"],
  "failure_type": "multi_party_same_speaker_collapse",
  "expected_property": "不能把整组问答全部标成同一人",
  "notes": "只记录通用失败类型，不把具体作品规则写入代码"
}
```

评估目标不是固定某本书的唯一答案，而是检查增强流程是否能发现风险：

- 是否生成 `consistency_issues`。
- 是否把明显冲突标注送入 repair。
- 是否减少高置信错误。
- 是否没有把所有不确定项都粗暴降成 review。

## 9. 分阶段落地

### Phase 1：文档和 prompt 设计

- 完成本文件。
- 明确新增任务 JSON schema。
- 确认是否接受更多模型调用换准确率。

### Phase 2：最小代码实现

- 新增 `chunk_consistency_review` 任务。
- 新增 `repair_targets`，让 repair 覆盖高置信 known 风险项。
- 输出 `consistency_issues.jsonl`。
- 不改实体系统大结构。

这是最优先的阶段，因为它直接解决“错得很自信所以不会被修”的问题。

### Phase 3：证据抽取和轮次结构

- 新增 `turn_structure`。
- 新增 `narration_evidence`。
- 将 Task A/D 改成候选裁决模式。

这阶段会增加 prompt 数量，但对复杂多人对话帮助最大。

### Phase 4：实体 alias scope

- 给实体卡增加 alias scope。
- 区分 `same_character`、`same_scene_role`、`same_group_type`。
- 避免把局部职能称呼全局化。

### Phase 5：评估闭环

- 建立 issue bad case set。
- 用 `--cache-only` 或小范围重跑对比改动前后。
- 记录 high-confidence error、review_count、repair_count、consistency_issue_count。

## 10. 建议优先级

最高优先级：

1. 新增 `chunk_consistency_review`。
2. repair 覆盖高置信 known 风险项。
3. 输出 `consistency_issues.jsonl` 方便人工看。

第二优先级：

1. 新增 `turn_structure`。
2. 新增 `narration_evidence`。
3. 初标注改为候选证据裁决。

第三优先级：

1. 实体 alias scope。
2. bad case set。
3. 统计报表。

这个顺序能先修当前最明显的流程漏洞，再逐步把模型的阅读过程拆得更像人类。

## 11. 不做的事情

- 不把 issue #5 中的具体角色、地点、称谓写进 Python 规则。
- 不用正则或关键词直接决定 speaker。
- 不把所有连续同一 speaker 都判为错误。
- 不为了降低 review_count 牺牲准确率。
- 不要求一次模型调用解决整段或整卷。

## 12. 最终建议

我建议下一次代码实现从 Phase 2 开始：保留现有 V2 主流程，只在 `global_summary` 和最终写入 annotations 之间插入 `chunk_consistency_review`，并让 repair 处理 review 发现的高风险项。

这样改动面相对小，但能直接覆盖 issue #5 中最关键的失败模式：模型已经给出高置信 `known`，但上下文和对话轮次明显不支持。后续再逐步引入 `turn_structure` 和 `narration_evidence`，把标注从“直接猜 speaker”升级为“候选证据裁决”。
