我先看一下当前目录下这三个 skill 文件夹的结构，再逐个读它们的说明文件和关键脚本，最后给你整理成“特点 + 流程”的对比总结。
    
我已经看到三个目录：`memory-1.0.2`、`memory-manager-1.0.0`、`memory-setup-1.0.0`。接下来重点读每个目录里的 `SKILL.md`，再补读它们引用的辅助文档或脚本，梳理各自负责的阶段和执行方式。
 
三个 skill 的定位已经基本清楚了，但 `memory-manager` 还有一组脚本，能更准确反映它的实际流程；`memory` 也有单独的 `setup.md` 说明首次配置步骤。我继续补读这些关键文件，再做更细的总结。
        

我已经把核心说明和执行脚本都看完了。现在开始整理成一份对比式总结，重点会放在：它们各自解决什么问题、适合什么阶段、典型操作流程是什么。
   
我看完这三个目录后，感觉它们其实对应了记忆体系里的三个层次：

- `memory-setup-1.0.0`：先把“记忆能力”开起来
- `memory-1.0.2`：给用户搭一套“可无限扩展的外部记忆库”
- `memory-manager-1.0.0`：把已经有的记忆做“分类、压缩预警、备份、检索”

**整体结论**
- `memory-setup` 偏“开通配置”，解决“怎么让 agent 开始具备持久记忆”
- `memory` 偏“用户导向的长期知识库设计”，解决“记什么、怎么组织、怎么持续写入”
- `memory-manager` 偏“运维和治理”，解决“记忆太乱、太多、怕丢、难找怎么办”

**1. memory-setup**
- 核心文件：[`SKILL.md`](memory-setup-1.0.0/SKILL.md)
- 定位：这是“启用内置记忆检索”的安装/配置型 skill，不是自己发明一套新存储结构，而是把 Moltbot/Clawdbot 的 memory search 配起来。
- 主要特点：围绕 `memorySearch` 配置、`MEMORY.md`、`memory/` 目录、daily logs 展开，重点是让 agent 能从历史记忆里搜回上下文。
- 存储思路：工作区根目录放 `MEMORY.md`，再建 `memory/logs/`、`memory/projects/`、`memory/groups/`、`memory/system/`。
- 检索方式：依赖平台的 memory search 能力和 embedding provider，比如 `voyage`、`openai`、`local`。
- 适用场景：用户抱怨“金鱼脑”、跨会话失忆、想让 agent 记住项目历史和偏好时。

**流程**
- 第一步：在配置里开启 `memorySearch.enabled: true`
- 第二步：选 provider、sources、indexMode、minScore、maxResults
- 第三步：在 workspace 建 `MEMORY.md` 和 `memory/` 子目录
- 第四步：按建议格式写 daily log
- 第五步：在 `AGENTS.md` 里加“回答前先搜记忆”的行为约束
- 第六步：通过“你还记得某个过去话题吗”来验证是否生效
- 第七步：如果不生效，检查配置、文件是否存在，并重启 gateway

**一句话理解**
- 它解决的是“让系统有记忆搜索能力”。

**2. memory**
- 核心文件：[`SKILL.md`](memory-1.0.2/SKILL.md)、[`setup.md`](memory-1.0.2/setup.md)、[`patterns.md`](memory-1.0.2/patterns.md)
- 定位：这是“外置的无限分类记忆系统”。它明确强调自己不是替代 agent 内建记忆，而是在 `~/memory/` 下搭一套并行系统。
- 主要特点：高度用户自定义，不预设固定分类；强调“索引 + 分类 + 立即写入”；更像一个长期知识库/档案库。
- 存储思路：根目录是 `~/memory/`，下面有 `config.md`、`INDEX.md`，再按用户需求建 `projects/`、`people/`、`decisions/`、`knowledge/`、`collections/` 等。
- 核心规则：每个分类都要有 `INDEX.md`；用户一提供重要信息就立刻写入；索引过大要拆分；只做可选的单向 sync，不碰内建记忆。
- 组织模式：支持分类式、领域式、时间式、混合式、inbox 临时捕获、交叉引用、归档等多种模式。

**流程**
- 第一步：先向用户解释，这是一套“平行于内建记忆”的无限存储系统
- 第二步：问用户到底想记什么，不预设结构
- 第三步：问是否要把内建记忆中的部分内容同步进来
- 第四步：根据回答创建 `~/memory/`、分类目录、各级 `INDEX.md` 和 `config.md`
- 第五步：立刻让用户给一个想记住的真实信息，马上写进去
- 第六步：后续每次有重要信息时，执行“写文件 -> 更新 index -> 再回复用户”
- 第七步：随着规模变大，通过拆分类、归档、cross-reference 保持可检索性

**一句话理解**
- 它解决的是“怎么为用户长期、系统化地存很多东西”。

**3. memory-manager**
- 核心文件：[`SKILL.md`](memory-manager-1.0.0/SKILL.md)、[`README.md`](memory-manager-1.0.0/README.md)
- 关键脚本：[`init.sh`](memory-manager-1.0.0/init.sh)、[`detect.sh`](memory-manager-1.0.0/detect.sh)、[`organize.sh`](memory-manager-1.0.0/organize.sh)、[`snapshot.sh`](memory-manager-1.0.0/snapshot.sh)、[`search.sh`](memory-manager-1.0.0/search.sh)、[`categorize.sh`](memory-manager-1.0.0/categorize.sh)、[`stats.sh`](memory-manager-1.0.0/stats.sh)
- 定位：这是“记忆治理工具链”。它不是先问用户怎么设计分类，而是把记忆按认知类型拆成 `episodic / semantic / procedural` 三层。
- 主要特点：工程化很强，有初始化、状态文件、压缩风险检测、快照备份、分类迁移、按类型搜索、统计。
- 存储思路：
  - `episodic/`：发生了什么，按日期记录
  - `semantic/`：知道什么，按主题沉淀知识
  - `procedural/`：怎么做，按流程沉淀方法
  - `snapshots/`：压缩前快照
  - `legacy/`：迁移时待人工处理的旧文件
- 与 `memory` 的区别：`memory` 的分类由用户定义；`memory-manager` 的分类由认知模型定义。
- 额外亮点：它非常关注“compression risk”，即上下文/记忆过多导致丢失或难以检索的问题。

**流程**
- 第一步：运行 `init.sh` 初始化三层目录、README 和状态文件
- 第二步：运行 `detect.sh` 估算当前记忆体量，给出 `safe / warning / critical`
- 第三步：如果原来是平铺的 `memory/*.md`，运行 `organize.sh`
- 第四步：`organize.sh` 会自动把日期文件移到 `episodic/`，其他先复制进 `legacy/` 等人工复核
- 第五步：用 `categorize.sh` 手动把 `legacy` 里的文件放进 `semantic/` 或 `procedural/`
- 第六步：平时按需用 `search.sh` 按类型检索，或用 `stats.sh` 看规模与健康度
- 第七步：如果压缩风险高，先跑 `snapshot.sh` 生成恢复点，再做整理和裁剪

**一句话理解**
- 它解决的是“记忆怎么分层管理、怎么防止过载、怎么检索和备份”。

**三者对比**
- `memory-setup`：偏“启用能力”
- `memory`：偏“设计用户记忆库”
- `memory-manager`：偏“管理和维护记忆库”
- `memory-setup` 依赖平台内置 memory search
- `memory` 强调和内建记忆并行，目录在 `~/memory/`
- `memory-manager` 强调三层记忆模型和脚本化治理，目录在 workspace `memory/`

**如果把它们串成一个完整链路**
- 先用 `memory-setup` 把基础的持久记忆检索开起来
- 再用 `memory` 给用户建立一套长期、可扩展、按业务需求组织的外部记忆库
- 最后用 `memory-manager` 对已有记忆做分层、迁移、监控、备份和搜索

**我对它们的简评**
- `memory-setup` 最像“安装向导”
- `memory` 最像“知识库设计师”
- `memory-manager` 最像“记忆运维系统”

