# EntroFlow 当前研究方法汇报稿

> 本文档依据当前仓库中的代码、研究范围文档和实验 artifacts 整理。
> 代码实现优先于较早的描述性文档。凡仓库没有提供可靠出处或实验支持的内容，均明确标注为“需要进一步确认”。

## 0. 当前材料判断

仓库已经足够整理“当前方法”汇报，但不足以支持最终 novelty claim 或完整外部文献综述。仓库没有正式论文引用清单，因此 Shannon entropy、semantic entropy、MuSiQue、AutoGen、KMeans 等方法的原始出处需要进一步 literature search。

代码与文档存在一处版本差异：早期文档描述为“entropy anomaly -> repair”，而当前 `EntropyPolicy` 实际先计算 `contract_violation_score`；只有没有明显 handoff-contract violation 时，才使用 semantic entropy 或 entropy delta。当前代码也已经可以选择 `EXPAND_EVIDENCE`，因此文档中“主分支尚未选择它”的表述已经过时。

依据：[research_scope.md](research_scope.md)、[entropy_repair_mechanism.md](entropy_repair_mechanism.md)、[routing.py](../src/entroflow/routing.py)。

## 1. Research Problem

### Research Problem

在固定 LLM backbone、固定 MuSiQue 数据划分、固定初始 AutoGen 工作流和固定推理预算下，多智能体问答系统出现错误时，能否通过运行时观察 collaboration trace，识别一个“首个可修复的失败位置”，并选择一个局部 repair action，提升答案恢复率，同时控制额外推理成本？

研究对象不是新的 QA 模型，而是一个运行时 workflow plug-in。AutoGen 负责 agent 执行、消息传递和工具调用；EntroFlow 负责读取结构化通信轨迹，并决定是否提出受限修复动作。

当前研究目标被限定为 `first recoverable failure point`，而不是 `true causal error`：该位置需要通过独立审计确认，且指定 repair 有机会改变下游答案。

### Research Objective

1. 设计一个不读取 gold answer 的运行时诊断器。
2. 验证 EntroFlow 是否能在相同预算下获得更高的 paired net rescue：

   \[
   \text{net rescue} = \text{rescued examples} - \text{regressed examples}
   \]

评价不能只看总体 accuracy，还要报告 rescue rate、regression rate、额外 token、延迟和 intervention cost。

### Main Challenge

1. 失败可能发生在 Planner、Evidence、Reasoner 或 Aggregator 的不同位置。
2. 通信语义发散不一定代表错误，也可能只是多个合理候选。
3. 只有 action 标签而没有真实上下文变化，不能算真正实施了 repair。
4. repair 可能救回错误样本，也可能破坏原本正确的样本。
5. 不同 hop 数、角色和通信边的 entropy 分布可能不同，固定阈值未必可迁移。

### Current Hypothesis / Intuition

当前代码体现的是一个两阶段假设：

> entropy/change signal 用于发现通信不稳定性；handoff-contract signal 用于判断具体丢失了哪个约束或 bridge relation；repair action 针对丢失的约束进行局部修复。

这比“直接修复最高 entropy 的角色”更窄，也更容易被实验 falsify。

## 2. Overall Pipeline

```text
MuSiQue question + passages
        |
        v
Planner: decompose into ordered multi-hop constraints
        |
        v
Evidence: retrieve bridge entities and cited evidence chain
        |
        v
Structured RoleResponse
(content, status, evidence_ids, token count, embedding)
        |
        +-----------------------------+
        |                             |
        v                             v
Handoff-contract score       Rolling / sampled semantic entropy
(status + unresolved markers  H_norm and delta
 + missing evidence)
        |                             |
        +-------------+---------------+
                      |
                      v
Runtime routing decision
(no repair / retry / validation / bypass / expansion / cross-check ...)
                      |
                      v
Change next role + visible history + repair instruction
                      |
                      v
Reasoner / Validator / Aggregator
                      |
                      v
Answer
                      |
                      v
EM, token-F1, cost, latency,
rescue/regression, paired evaluation
```

角色和静态路径见 [`routing.py`](../src/entroflow/routing.py) 与 [`agents.py`](../src/entroflow/agents.py)。

## 3. Step-by-Step Method

### Step 1: 固定多智能体 QA 工作流

**目的**

建立可比较的初始 workflow，使后续差异主要来自 runtime controller，而不是 backbone 或 prompt 改变。

**输入**

MuSiQue 问题、候选 passages，以及固定模型配置。

**具体方法**

静态路径主要是：

```text
Planner -> Evidence -> Reasoner -> Aggregator
                       ^
                   Validator（repair 时插入）
```

角色使用结构化 JSON 输出：

```json
{"content": "...", "answer": "...", "status": "ok", "evidence_ids": [0, 1]}
```

Planner 输出有序 hop、subject、relation 和 expected object type；Evidence 返回有序证据链和 passage 引用；Reasoner 依据引用证据逐跳推理；Aggregator 输出最终 answer span。

**选择原因**

这样可以把多跳问答拆成可审计的 handoff contracts：Planner 传递约束，Evidence 传递桥接证据，Reasoner 传递支持充分的推理链，Aggregator 传递最终答案。

**方法依据**

这是仓库中的系统设计。仓库没有给出这些 prompt 设计的外部原始论文依据，需要进一步确认。

**替代方案**

| 方法 | 优点 | 缺点 | 当前选择 |
|---|---|---|---|
| 单个 LLM 直接回答 | 简单、成本低 | 无法定位中间失败 | 不符合 trace localization 目标 |
| 固定多 agent workflow | 可审计、变量可控 | 可能继承静态拓扑缺陷 | 当前基线 |
| 动态生成任意 workflow | 灵活 | 难以公平比较和归因 | 当前不采用 |

**潜在问题**

- 固定拓扑可能人为限制性能；
- prompt 本身会影响失败分布；
- Validator 不在静态 baseline 中，因此 repair 同时改变了拓扑和信息可见性。

**验证方式**

固定模型、prompts、数据划分、最大步数、action library 和 token/call budget。

### Step 2: 结构化 trace 与 embedding 观测

**目的**

把自然语言通信转化为 runtime controller 可以读取的信号。

**输入**

每个角色的 `RoleResponse`，包括 `role`、`status`、`content`、`evidence_ids`、token count、semantic entropy / cluster count 和 trace metadata。

**具体方法**

消息文本通过固定 embedder 转为向量。每条有向通信边分别维护历史，例如：

```text
planner_to_evidence
evidence_to_reasoner
reasoner_to_aggregator
```

**选择原因**

边级观测比整条 workflow 的全局分数更适合回答“哪一条 handoff link 出现了不稳定”。

**方法依据**

这是 [`research_scope.md`](research_scope.md) 中定义的系统边界。embedding 模型和边级设计的外部文献依据需要进一步确认。

**潜在问题与验证**

embedding 模型、距离度量和文本格式都会影响聚类结果。需要更换 embedding 配置，报告 entropy、触发率和 repair 结果是否稳定。

### Step 3: Rolling Semantic Entropy

**目的**

检测一条通信边上的语义内容是否出现分散或突变。

**输入**

某条边最近 (W) 条消息的 embedding：

\[
z_1,z_2,\ldots,z_W
\]

当前默认参数：`window_size=32`、`min_samples=8`、`max_clusters=4`、KMeans `random_state=42`。

**具体方法**

对窗口 embedding 聚类，得到 cluster label (c_i)。第 (k) 个 cluster 的比例为 (p_k)：

\[
H=-\sum_{k=1}^{K}p_k\log p_k
\]

归一化后：

\[
H_{norm}=\frac{-\sum_{k=1}^{K}p_k\log p_k}{\log K}
\]

变化量：

\[
\Delta_t=H_{norm,t}-H_{norm,t-1}
\]

这里的 entropy 是 embedding cluster distribution 的 semantic dispersion，不是 token-level predictive entropy、logits uncertainty 或 gold-aware error probability。

**直觉**

- (H_{norm}) 高：窗口存在多个语义簇；
- (Delta>0)：通信语义相对上一时刻更分散；
- (Delta<0)：语义分布回落或趋于集中。

**选择原因**

当前 working hypothesis 是通信不稳定性可能早于最终答案错误暴露，因此尝试用运行时 trace 进行早期干预。

**方法依据**

Shannon entropy 和 KMeans 是经典工具，但仓库没有提供原始论文引用。把 embedding cluster entropy 用作通信诊断信号是当前研究设计，不能包装成已经被仓库验证的文献结论。

**替代方案**

| 方法 | 优点 | 缺点 | 当前选择 |
|---|---|---|---|
| 绝对 entropy | 直观 | 混淆正常多样性和异常 | 作为辅助信号 |
| entropy delta | 接近突变时刻 | 对窗口和前一状态敏感 | 当前主要信号 |
| token predictive entropy | 更接近概率不确定性 | 需要 logits / logprob | 当前未采用 |
| transition surprisal | 可描述通信变化 | 需要更多历史建模 | 后续对照 |
| source-target alignment | 直接衡量 handoff 对齐 | 需要定义对齐目标 | 后续对照 |

**潜在问题**

- cluster 数随 unique vector 数和窗口状态变化；
- KMeans 有随机性；
- `threshold=0.15` 没有 held-out calibration；
- 高 entropy 可能只是多个合理候选。

**验证方式**

进行窗口、threshold、embedding、clustering、absolute entropy、delta 和 transition surprisal 的 ablation，并在 held-out split 上锁定参数。

### Step 4: Handoff-Contract Diagnosis

**目的**

区分“存在多个合理候选”和“Evidence 没有完成桥接关系 / Reasoner 使用了无证据支持的 hop”。

**输入**

最近一个 `RoleResponse` 的 `status`、`content`、`evidence_ids` 和 `role`。

**具体方法**

当前代码计算：

\[
S_{contract}=\max(S_{status},S_{marker},S_{missing-evidence})
\]

`status` 为 `unparsed`、`insufficient`、`uncertain`、`invalid` 或 `retry` 时提供结构化分数；正则表达式搜索 `missing`、`unsupported`、`uncertain`、`contradiction`、`insufficient` 等 marker；Evidence 或 Reasoner 出现 unresolved marker 且没有 evidence IDs 时提高分数。

当 (S_{contract}\ge0.6)，且角色为 Evidence 或 Reasoner 时，优先走 contract repair。

**动作映射**

- Evidence contract violation -> `EXPAND_EVIDENCE`；
- Reasoner contract violation -> `INSERT_VALIDATION`。

**选择原因**

contract signal 更接近“丢失了什么约束”，因此比单独选择最高 entropy 角色更适合决定 repair 类型。

**方法依据**

这是当前 heuristic / engineering design。它不是 gold label，也不是因果证明。

**已有证据**

`artifacts/entropy_pilot_20/contract_evaluation.json`：20 条 audit 中 16 条命中，探索性 accuracy 为 0.80。但 truth distribution 为 Evidence 17、Aggregator 3，majority-role baseline 为 0.85，因此该结果不能声称优于 baseline。

**替代方案**

| 方法 | 优点 | 缺点 | 当前选择 |
|---|---|---|---|
| 只用 entropy | 通用、不需结构化输出 | 区分不了合理分歧与约束丢失 | 不单独采用 |
| 只看 status | 简单 | 依赖模型正确填写 status | 作为组成部分 |
| NLI / entailment | 语义更直接 | 额外模型与成本 | 尚未作为主路径 |
| status + marker + 引用 | 低成本、可审计 | lexical bias | 当前方案 |

**潜在问题**

marker 可能 false positive；模型也可能不报告失败；当前角色分布不平衡；Evidence 先验可能造成偏置。

**验证方式**

使用 blinded audit、held-out calibration、balanced role strata、status-only 对照和 marker removal ablation。

### Step 5: 触发与 Repair Action

**目的**

决定是否干预，以及采用哪一个局部 repair。

**实际优先级**

```text
if contract_score >= 0.6:
    contract-specific repair
elif semantic_entropy exists:
    use entropy / delta
else:
    update rolling edge entropy
```

entropy 分支主要使用：

\[
anomalous=ready\land(H_{norm}\ge0.6\ \lor\ |\Delta|\ge0.15)
\]

严重程度为：

\[
severe=|\Delta|\ge2\times0.15
\]

| 条件 | 动作 |
|---|---|
| Evidence，(Delta>0)，severe | `INSERT_VALIDATION` |
| Evidence，(Delta>0)，非 severe | `CROSS_CHECK` |
| Evidence，(Delta<0) | `REORDER_VALIDATION` |
| Reasoner，anomaly，severe | `EVIDENCE_BYPASS` |
| Reasoner，anomaly，非 severe | `COMPRESS_CONTEXT` |

repair 同时改变 `next_role`、`visible_roles`、`RoleRequest.metadata["repair_action"]`、接收 agent 上下文以及 trace 中的 diagnosis、target、instruction 和 confidence。

**关键限制**

当前规则是手工 decision table。尚未证明每个 action 都必要，也尚未通过 action-level ablation 区分 entropy 判断和 contract 判断各自的贡献。

### Step 6: Evaluation

**比较对象**

1. Static AutoGen；
2. Random Repair；
3. Trace-only Repair；
4. EntroFlow。

**指标**

EM、token-level F1、total tokens、mean token cost、latency、intervention cost、rescue rate、regression rate、paired net gain，以及 2/3/4-hop 和 retrieval dead-end 分层结果。

**配对定义**

- baseline 错、candidate 对：rescue；
- baseline 对、candidate 错：regression；
- 两者相同：unchanged。

## 4. 当前仓库结果

### 4.1 20 条 paired repair pilot

| Controller | Accuracy | Macro F1 | 总 token |
|---|---:|---:|---:|
| Static | 35.0% | 40.33% | 91,758 |
| Random Repair | 40.0% | 45.33% | 95,403 |
| Trace-only | 30.0% | 35.25% | 146,874 |
| Entropy Repair | 30.0% | 33.67% | 91,613 |

| Controller | Rescued | Regressed | Net gain | Baseline failure rescue |
|---|---:|---:|---:|---:|
| Random Repair | 1 | 0 | +1 | 7.69% |
| Entropy Repair | 0 | 1 | -1 | 0% |

来源：[paired_repair_evaluation.json](../artifacts/paired_repair_evaluation.json)。

**当前可说的结论：**当前 20 条 pilot 不支持“Entropy Repair 有效”；它没有产生 rescue，并出现 1 个 regression。

### 4.2 Entropy localization pilot

普通 entropy signal：assignment entropy 3/20、entropy delta 5/20、semantic surprise z 3/20；majority-role baseline 为 17/20 = 85%。conditional / historical entropy 的最好结果为 9/20 或 8/20，仍低于 majority baseline。

来源：[evaluation.json](../artifacts/entropy_pilot_20/evaluation.json)、[conditional_evaluation.json](../artifacts/entropy_pilot_20/conditional_evaluation.json)、[historical_evaluation.json](../artifacts/entropy_pilot_20/historical_evaluation.json)。

**当前可说的结论：**单独 entropy 不能可靠定位 first causal failure point，这正是方法转向 contract-aware two-stage diagnosis 的原因。

### 4.3 Contract diagnostic pilot

`contract_evaluation.json` 报告 20 个 audit、16 个 hits、0.80 exploratory accuracy，并明确要求 held-out calibration。由于 majority baseline 为 0.85，该结果只能说明“更有希望”，不能说明已经验证有效。

### 4.4 测试状态

当前环境执行 `pytest -q` 未能完成收集，原因包括缺少 `autogen_core`、Python 3.7 的 `typing` 兼容性以及 SciPy 版本缺少 `jenshannon`。因此不能在组会上说“测试已通过”。

## 5. Why This Pipeline Might Work

```text
多跳 QA 错误通过 agent handoff 传播
        -> 最终答案无法定位错误
        -> 结构化 trace 暴露 status、evidence IDs 和内容
        -> entropy 提供通信不稳定性信号
        -> contract check 判断丢失的 bridge / constraint
        -> 局部 repair 改变后续角色、可见历史和 instruction
        -> 可能 rescue baseline failure
```

理论支持较强的是 Shannon entropy、聚类分布和 paired evaluation；经验性最强的是“entropy delta 对应失败转折点”；最需要验证的是 contract marker 的泛化，以及 repair 是否真正改善答案。

## 6. 组会 PPT 讲稿

### 第 1 页：问题定义

```text
Runtime repair for multi-agent multi-hop QA
Fixed backbone / split / workflow / budget
Target: first recoverable failure point
```

**口头讲：**“我们不是重新训练 QA backbone，而是研究固定多智能体工作流中的运行时修复：只根据通信轨迹寻找首个可修复位置，并在不读取 gold 的情况下选择局部 repair。这里不把 recoverable point 等同于真实因果错误。”

### 第 2 页：固定工作流

```text
Planner -> Evidence -> Reasoner -> Aggregator
                       ^
                   Validator
```

**口头讲：**“为了隔离 controller 的作用，我们固定 backbone、prompt、数据划分和预算。Planner 分解 hop，Evidence 返回引用链，Reasoner 依据证据推理，Validator 只在 repair 时插入。”

### 第 3 页：两阶段诊断

```text
Stage 1: semantic instability (H_norm, ΔH)
Stage 2: handoff-contract violation
```

**口头讲：**“Pilot 表明单独 entropy 定位较弱，因此当前改成两阶段：entropy 提示通信不稳定，contract check 判断丢失了哪个约束并选择 repair。”

### 第 4 页：Entropy 定义

\[
H_{norm}= -\frac{\sum p_k\log p_k}{\log K},
\qquad \Delta H_t=H_t-H_{t-1}
\]

**口头讲：**“这里不是 logits uncertainty，而是消息 embedding 聚类后的语义分布熵；它是否能诊断通信失败仍是 working hypothesis。”

### 第 5 页：Repair action

```text
Evidence contract violation -> EXPAND_EVIDENCE
Reasoner contract violation -> INSERT_VALIDATION
Entropy anomaly -> CROSS_CHECK / REORDER / BYPASS / COMPRESS
```

**口头讲：**“Repair 会真实改变 next role、visible history 和 instruction，而不只是记录标签。Evidence 丢失桥接关系时扩展证据；Reasoner 出现 unsupported hop 时插入 Validator。”

### 第 6 页：当前结果与限制

- Entropy localization：5/20 best signal；
- Contract diagnostic：16/20 exploratory；
- Paired entropy repair：0 rescue, 1 regression；
- 当前结论：尚未证明有效。

**口头讲：**“目前还不能支持 EntroFlow 已经有效。Entropy repair 没有 rescue，并出现一次 regression；contract signal 也受小样本、类别不平衡和未做 held-out calibration 的限制。下一步是 component 和 action-level ablation。”

## 7. 已有依据与潜在创新

| 方法组成 | 来源 | 已有工作做到什么 | 我这里做了什么 | 是否可能构成创新 | 还需要什么证据 |
|---|---|---|---|---|---|
| 多 agent MuSiQue QA | 仓库系统设计 | 固定 AutoGen workflow | runtime plug-in 监控 | 需进一步 literature search | 与已有 agent workflow 对比 |
| Shannon entropy | 经典信息论 | 用分布衡量不确定性 | 用 embedding cluster distribution 定义通信 entropy | 单独不构成创新 | 原始引用和定义验证 |
| Rolling edge entropy | 当前工程实现 | 轨迹动态监控属于已有思想范畴 | 按 directed edge 维护窗口 | 可能是系统组合贡献 | 与 trace monitoring 对比 |
| Entropy delta | 当前 hypothesis | 变化检测是常见思想 | 用 ΔH 触发局部 repair | 需进一步 literature search | threshold / window calibration |
| Handoff-contract score | 当前 heuristic | status/evidence validation 有相关思想 | 组合 status、marker、引用信息 | 可能是方法组合贡献 | held-out、balanced audit、ablation |
| Contract-specific repair | 当前设计 | validation/retry/evidence expansion 并非全新 | 根据丢失约束类型选择 repair | 不宜单独 claim novelty | matched-cost action ablation |
| Paired rescue/regression | 评估协议 | 配对比较是合理评估方式 | 将 rescue/regression/token cost 作为主结果 | 不是算法创新 | 更大规模 paired evaluation |
| Two-stage entropy + contract | 当前 working hypothesis | 具体组合出处未确认 | entropy 发现不稳定，contract 定位约束 | 可能是潜在贡献 | literature search 和 held-out gain |

## 8. 严格导师追问

| 追问 | 安全回答 | 需要补充 |
|---|---|---|
| 为什么用 embedding entropy，而不是 logits entropy？ | 当前接口提供消息和 embedding；二者不能视为等价 | predictive entropy / logprob 对照 |
| 高 entropy 为什么代表失败？ | 不一定；当前只把它作为 instability signal | entropy-only 与 contract-aware 对照 |
| 为什么 threshold 是 0.15？ | 当前是工程参数，未证明最优或可迁移 | development calibration + held-out evaluation |
| contract marker 是否有 lexical bias？ | 有可能；它只是 heuristic diagnostic | marker removal、paraphrase robustness |
| 16/20 是否优于 baseline？ | 不是；80% 低于 majority baseline 85% | balanced held-out audit |
| repair 是否真正改变输入？ | 代码修改角色、可见历史和 instruction | context diff、instruction execution audit |
| 为什么不总是插入 Validator？ | 成本更高，且无法验证 controller 的选择价值 | Always-validate baseline |
| improvement 来自 entropy 还是 contract？ | 目前无法区分 | Static / Entropy-only / Contract-only / Combined |
| 是否存在 information leakage？ | runtime 不读 gold，但仍需完整审计 | 检查脚本、metadata 和数据划分 |
| 为什么最多一次 repair？ | 为控制预算和归因复杂度 | 多次 repair 的预算敏感性实验 |
| 不同 artifact 为什么样本量不同？ | 不同规模实验不能混合比较 | 统一 paired evaluation set |
| 当前没有收益，为什么继续？ | 当前结果否定了 entropy-only 实现，支持转向更窄的 contract-aware hypothesis | 校准、action audit、component ablation |

## 9. 会前急救版

### 30 秒版本

“我们研究的是固定 LLM backbone 下多智能体多跳问答的运行时修复。系统先把 Planner、Evidence、Reasoner 之间的消息记录成结构化 trace，并在通信边上计算 embedding-based semantic entropy 及其变化，用来发现通信不稳定。由于 pilot 显示单独 entropy 不能可靠定位错误，当前进一步加入 handoff-contract 检查，根据 status、未解决标记和证据引用判断是 Evidence 丢了 bridge relation，还是 Reasoner 出现 unsupported hop，然后选择 evidence expansion 或 validation 等受限 repair。最终通过 paired rescue、regression、准确率和额外 token 评估。当前结果还不能证明方法有效。”

### 2 分钟版本

“问题是多智能体多跳问答中的错误会沿着 Planner 到 Evidence、Reasoner、Aggregator 的 handoff 传播，但最终答案错误本身不能告诉我们错误最早发生在哪里。我的目标不是训练一个新的 QA backbone，而是在固定模型、prompt、数据划分和预算下，加入 runtime plug-in，尝试找到首个可修复的失败位置。

当前 pipeline 是先让 Planner 把问题拆成有序 hop，Evidence 搜索包含 bridge entity 的完整证据链，Reasoner 基于引用证据逐跳推理，Aggregator 输出答案。每条角色响应都结构化记录 status、evidence IDs、token count 和 embedding。

诊断部分是两阶段。第一阶段对每条通信边维护 rolling embedding window，用聚类分布计算归一化 Shannon entropy，并看 entropy delta，判断消息是否发生语义发散。第二阶段检查 handoff contract，例如 Evidence 是否报告缺失 hop、Reasoner 是否有 unsupported relation 或缺少证据引用。实际代码中 contract signal 优先于 entropy，因为 pilot 表明单独 entropy 定位能力比较弱。

触发 repair 后，系统改变下一角色、可见历史和具体 instruction。例如 Evidence contract violation 触发 evidence expansion，Reasoner contract violation 插入 Validator。我们不把 trace 中出现 repair_action 标签直接视为修复成功，而是要求检查接收 agent 的上下文是否真的发生改变。

当前 20 条 paired pilot 中，entropy repair rescue 为 0，并有 1 个 regression；contract diagnostic 在 20 条探索性审计中命中 16 条，但没有超过 majority-role baseline，也没有 held-out calibration。因此目前最准确的结论是：two-stage contract-aware runtime repair 是 working hypothesis，下一步需要通过 component ablation、阈值校准和更大规模 paired evaluation 验证。”

### 一句话贡献

> 提出并评估一种面向固定多智能体多跳问答工作流的 contract-aware runtime repair 机制，用通信语义变化发现不稳定性，再针对丢失的 handoff constraint 选择局部修复；其有效性目前仍需 held-out paired experiments 验证。

### 今天最危险的 5 个问题

1. **16/20 是否真的优于 baseline？** 当前没有；majority baseline 为 85%。
2. **Entropy repair 没有 rescue 还 regression，为什么继续？** 当前实现未验证成功，后续重点是 contract-aware policy 和 ablation。
3. **创新和已有 semantic entropy / self-correction 的区别？** 尚未完成系统 literature search，不能直接 claim novelty。
4. **如何证明 repair 改变了模型行为？** 需要 context diff、instruction execution audit 和 action ablation。
5. **0.15 与 window size 32 的依据？** 当前是工程设定，不是已验证最优参数，需要校准和敏感性分析。
