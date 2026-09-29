# 面向多智能体工作流的机会条件化诊断与组件级效用分解

## 摘要

多智能体工作流中的最终错误可能由当前节点产生，也可能由上游错误传播而来。仅依据末端失败或组件失败频率进行归因，会混淆错误来源、当前恢复机会与最终输出位置。本文提出 EntroFlow：一种基于激活执行图、角色契约与可见输入的机会条件化诊断方法。该方法以 execution instance 为归因单位，分别记录节点是否执行、必要输入是否存在且可见、职责是否适用以及恢复信息是否充分，并据此判断组件是否拥有改变当前失败状态的机会。

实验从受控定位、真实干预和组件瓶颈三个层次评估 EntroFlow。HumanEval 的 108 个 compositional fault cases 中，Raw、Activation-only 和 Full Opportunity 的注入源定位率分别为 0%、53.7% 和 82.4%；Full Opportunity 的 inactive 与 no-opportunity false blame 均为 0。冻结诊断规则后，组合拓扑 C1 无需 amendment 即可运行。在自然 HumanEval-164 执行中，Full Opportunity 消除了 Activation-only 的 19 次无恢复机会干预，但最终为 1 rescue、1 regression，与 matched controls 相比没有净效用优势。进一步的 D/V/S/R/A 分解显示，live verifier 将 41/139 个官方正确候选误判为失败；18 个具有恢复机会的失败中仅 1 个被 Repair 修复；C1 的 Judge 在 150 个至少包含一个正确分支候选的样本中破坏了 85 个。Oracle-preserving aggregation 的离线 headroom 为 19 个正确答案，高于 oracle verifier 的 1 个。

这些结果把 diagnosis improvement 与 end-to-end utility 的差距定位到触发、修复和聚合环节。历史 Coordinator–Worker intervention 仍将 HumanEval pass@1 从 85.37% 提高到 92.07%（11 rescues、0 regressions；Holm-adjusted McNemar $p=0.00488$），说明诊断可以支持有效干预；新的 live/compositional 结果则表明，可靠归因本身不足以保证修复成功或候选保留。Omni-MATH 和 MuSiQue 的结果进一步显示，局部 failure structure 改善与最终准确率需要分别报告。本文据此将多智能体工作流优化表述为一个可审计链条：诊断、触发、选择、修复、聚合与配对效用评估。

**关键词：** 多智能体系统；机会条件化；激活执行图；错误传播；局部干预；候选保留

## 1. 引言

大语言模型驱动的多智能体系统通过规划、并行求解、验证、批判、聚合和修复等角色组织推理过程。结构化协作增加了可用的求解路径，也增加了错误传播路径：一个节点的失败可能是局部生成错误、错误路由、候选聚合破坏，也可能只是接收了无法恢复的上游状态。

末端错误并不等价于局部薄弱点。若 Planner 遗漏关键约束，后续 Solver 或 Aggregator 即使输出错误，也未必拥有恢复原任务的充分信息。反之，当 Aggregator 同时看到正确和错误候选却丢弃正确候选时，该节点具有明确的局部失败机会。反馈结构还引入时间维度：`verifier@0`、`repair@0` 和 `verifier@1` 是不同 execution instances，其可见证据与恢复机会不能按静态角色名合并。

EntroFlow 将归因建立在样本级激活执行图上。静态图描述允许的控制流和数据流；激活图记录实际执行实例、遍历边、迭代、可见输入、输出、路由和停止决策。机会判断由执行状态、输入可见性、节点契约和因果访问共同决定。该表示适用于 Sequence、Fork/Join、Select 和 bounded feedback，也支持这些结构的组合，而无需为每个 topology 编写独立归因规则。

本文进一步区分诊断质量与干预效用。真实工作流中的最终结果依次受到五个组件影响：

$$
\text{Diagnosis (D)}
\rightarrow \text{Verification/Trigger (V)}
\rightarrow \text{Selection (S)}
\rightarrow \text{Repair (R)}
\rightarrow \text{Aggregation (A)}.
$$

这一分解解释了两类表面矛盾的结果。历史 HumanEval Coordinator–Worker repair 在可靠 failure cohort 和丰富 contract evidence 下获得 11 个净 rescue；新的 live conditional pipeline 虽减少无机会干预，却因 verifier false trigger 和较低 repair success 未获得净提升；组合拓扑 C1 则主要受 Judge candidate destruction 影响。

本文的贡献如下：

1. 提出 execution-instance-level opportunity attribution，以激活状态、输入可见性、契约适用性与恢复信息刻画当前节点的因果机会。
2. 构建由 Sequence、Fork/Join、Oracle-Gated Select 和 bounded feedback 组成的可组合拓扑执行与受控 fault-injection 框架，并在冻结诊断后评估未调规则的组合拓扑 C1。
3. 在完整 HumanEval-164 上同时评估定位、干预选择、rescues/regressions、真实调用成本和 candidate preservation，展示诊断改善与自然任务效用之间的差距。
4. 用 D/V/S/R/A counterfactual replay 定位效用瓶颈：当前主要损失来自 verifier false trigger 与 aggregation destruction，Repair 成功率构成次要限制。
5. 结合 Omni-MATH 和 MuSiQue 的配对实验，比较结构机制改善、相对 actionability 与最终性能三种不同证据。

## 2. 问题定义

### 2.1 静态图与激活执行图

将工作流的静态可能图表示为：

$$
G_s=(V_s,E_s),
$$

其中节点由角色契约定义，边描述允许的数据流和控制流。对样本 $x$，实际运行产生激活执行图：

$$
G_a(x)=(V_a(x),E_a(x)).
$$

执行实例 $v_t\in V_a(x)$ 至少包含：

- `static_node_id` 与唯一 `execution_id`；
- `role`、`iteration` 与 `ordinal`；
- `activated`、`activation_reason` 与 `skip_reason`；
- 该实例可见的输入、输出与 provenance；
- 路由、停止、调用、token 与延迟信息。

未激活节点仍保留显式记录。`inactive_by_policy` 表示控制流选择未执行该节点；`missing_trace` 表示预期记录缺失。前者没有机会，后者属于执行或观测缺陷。

### 2.2 Opportunity

对执行实例 $v_t$，定义可观察机会分量：

$$
O(v_t)=f(A_t,P_t,V_t,C_t,R_t),
$$

其中：

- $A_t$：该 execution instance 是否激活；
- $P_t$：契约要求的输入是否存在；
- $V_t$：必要输入是否对该实例可见；
- $C_t$：节点契约是否适用于当前状态；
- $R_t$：若节点承担恢复职责，是否存在可行动的恢复信息。

归因输出同时保存各分量与 `opportunity_reason`。例如，Repair 只有在看到失败候选、任务契约和可行动 verifier feedback 时才拥有完整恢复机会。第二个 Verifier 检测到 Repair 仍然失败，并不自动使该 Verifier 成为失败来源。

### 2.3 Origin 与 Actionability

受控注入包含两个不同标签：

- **Origin**：最初被人为破坏的 execution instance；
- **Actionability**：当前执行状态中实际拥有恢复机会的位置。

当上游 Solver 被注入错误，而下游 Repair 获得完整错误证据和修复契约时，Origin 位于 Solver，当前 Actionability 可以位于 Repair。本文分别报告两类标签。Actionability re-label 由冻结的节点契约和 opportunity 字段构造，用于机制一致性分析；它不是独立人工语义 gold。

### 2.4 从诊断到效用

对同一组样本，干预效用使用配对转移衡量：

$$
N_{\text{net}}=N_{\text{rescue}}-N_{\text{regression}}.
$$

其中 rescue 表示基线错误而干预后正确，regression 表示基线正确而干预后错误。最终准确率、配对转移、置信区间、exact McNemar、调用量、tokens 与延迟共同描述效用。定位精度不作为最终性能的替代指标。

## 3. 方法

### 3.1 结构原语

当前实现包含以下原语：

- `Sequence(A,B)`；
- `Fork(A→{B1,…,Bk})`；
- `Join({B1,…,Bk}→C)`；
- `Select(condition,{path1,…,pathk})`；
- `Iterate(condition,body,max_rounds)`；
- `Terminate(condition)`。

P1–P4 分别实例化 Sequence、Fork/Join、Oracle-Gated Select 和最多两轮 bounded feedback。C1 组合 Fork、Join、Judge aggregation、conditional repair 与 bounded feedback。SharedState 与层级 C2 不在当前实验范围内。

P3 使用 official isolated evaluator 构造 Oracle-Gated Select，只验证条件激活与 routing fault attribution。它不评估 learned routing policy。

### 3.2 诊断消融

三种归因方法共享相同轨迹：

- **A0 Raw**：依据失败状态直接归因；
- **A1 Activation-only**：排除未激活实例；
- **A2 Full Opportunity**：同时使用激活、输入可见性、契约适用性和因果恢复信息。

A2 不读取 `topology_id` 或 fault label。Topology-specific 信息位于节点契约和执行 metadata 中。诊断规则在 primitive validation 后写入 manifest；C1 运行前验证 diagnosis code、opportunity rules、primitive schema 与 contract hashes。

### 3.3 受控故障

受控实验包含：上游生成破坏（F1）、错误路由（F2）、单分支破坏（F3）、Aggregator destruction（F4）、有充分恢复信息的 Repair fault（F5）和缺少恢复信息的 Repair activation（F6）。每个 case 分离保存注入 ground truth 与 diagnosis input，并在 evaluation 阶段连接。

F5/F6 构成机会条件化的关键对照：两者均激活 Repair，但只有 F5 提供失败候选、任务契约和可行动反馈。

### 3.4 自然执行与 matched controls

HumanEval 自然执行比较：

- B0 frozen initial Coder output；
- A0/A1/A2 conditional repair；
- random matched intervention；
- blind matched-call/token repair；
- diagnosis-only；
- 历史 Coordinator–Worker intervention；
- C1 parallel solvers、Judge 与 conditional repair。

V-live 仅使用工作流可见的 generated/public tests 与执行反馈；V-oracle 使用 official isolated evaluator，只用于离线分析和 upper bound。自然失败没有 source ground truth，本文报告 diagnosis distribution、intervention、rescue、regression 与 cost，并保留人工 audit artifact。

### 3.5 D/V/S/R/A 瓶颈分解

Phase II-B 优先复用已保存 artifacts，通过 component substitution 估计可识别 headroom：

- V-live 与 V-oracle；
- S-A2 与仅在受控标签上合法的 S-oracle；
- 已保存的 R-live；
- A-live、基于合法可见 evidence 的 deterministic preservation，以及 A-oracle-preserve。

A-oracle-preserve 在分支候选中存在官方正确代码时保留该候选，不生成新代码。它只衡量 candidate-preservation headroom。Phase II-B 使用 0 个新增 LLM 调用。

## 4. 实验设置

### 4.1 HumanEval

HumanEval 使用全部 164 个任务。初始 Coder candidate 由冻结 artifact 提供；新增 Verifier、Repair、Solver-B 与 Judge 事件记录 `cached | live | deterministic_tool` provenance。官方 evaluator 在隔离环境运行，不向 V-live 或 Repair 暴露 hidden tests。

Phase II protocol 与诊断 manifest 在 full execution 前冻结。最终 diagnosis hash 为 `1ea83d3c…23ff`，opportunity rule hash 为 `bf22617…0c8f`；C1 前后哈希一致，无 amendment。

### 4.2 Omni-MATH

Omni-MATH 使用 difficulty 1.5–3.5、Qwen3-8B、temperature 0、thinking disabled 与 `math_equivalence_v2` evaluator。主实验包含 3 个 seeds，每个 seed 300 个 matched samples，共 900 个 seed×sample pairs。G0 包含两个并行 solver、Critic、Refiner 与 Finalizer；G1 将第二 solver 改为基于第一解寻找反例的 adversarial second solver。

### 4.3 MuSiQue

MuSiQue 使用 dev split 的 300 个固定样本，覆盖 2/3/4-hop 问题。工作流为 Planner→Evidence→Reasoner→Aggregator。当前语义 root-cause gold 未完成，因而自然轨迹只报告 lexical/contract proxy 与 end-to-end paired outcomes。

## 5. 实验结果

### 5.1 Compositional controlled diagnosis

完整 cohort 包含 108 个适用的 task×fault cases；36 个结构上不适用的组合在预注册规则下排除。

| 方法 | Origin localization | Downstream false blame | Inactive false blame | No-opportunity blame | Mean blame distance |
|---|---:|---:|---:|---:|---:|
| A0 Raw | 0.0% | 46.3% | 53.7% | 95.4% | 4.86 |
| A1 Activation-only | 53.7% | 46.3% | 0.0% | 41.7% | 2.25 |
| A2 Full Opportunity | 82.4% | 17.6% | 0.0% | 0.0% | 1.09 |

A2 相比 A1 提高 28.7 个百分点，并消除 inactive 与 no-opportunity blame。F1/F3 中的部分 origin mismatch 来自后续 Repair 获得完整恢复信息：按 Origin 标签计为下游偏移，按当前 Actionability 则是有效恢复位置。基于冻结 contract 构造的 actionability label 与 A2 在 108 个 case 上一致；这一结果描述规则内部一致性，不构成独立语义准确率估计。

C1 在冻结诊断下完成，没有新增 topology-specific attribution rule，也没有 diagnosis amendment。该结果支持已实现原语组合上的 controlled diagnostic generalization。

### 5.2 HumanEval 历史 Coordinator intervention

历史 G0 为 140/164（85.37%）。Coordinator–Worker repair 仅作用于初始 Coder failures，达到 151/164（92.07%），相对 G0 为 11 rescues、0 regressions。paired bootstrap 95% CI 为 $[+3.05,+10.98]$ 个百分点，exact McNemar raw $p=0.00098$，Holm-adjusted $p=0.00488$。Matched-call random control 为 1 rescue、1 regression。

| 条件 | Correct | Pass@1 | Rescues | Regressions | Net |
|---|---:|---:|---:|---:|---:|
| G0 | 140 | 85.37% | — | — | — |
| Coordinator–Worker | 151 | 92.07% | 11 | 0 | +11 |
| RepairCritic→Debugger | 145 | 88.41% | 5 | 0 | +5 |
| Matched-call random | 140 | 85.37% | 1 | 1 | 0 |

Coordinator 同时获得 official failure report、Planner specification 与实现结果，并将 contract mismatch 转化为 Debugger 可执行的修复建议。139 个初始正确样本绕过 repair，因此 whole-workflow token ratio 为 0.742×；在 25 个 failure 的 repair path 内，token ratio 为 2.703×。这是一项对失败 cohort 增加定向计算的 intervention，而不是低成本 repair。

### 5.3 HumanEval live conditional intervention

Full HumanEval-164 的 T2 使用 V-live 触发最多两轮 Repair。主要结果如下：

| 条件 | Correct | Rescues | Regressions | Net | Interventions |
|---|---:|---:|---:|---:|---:|
| A0 Raw | 136 | 1 | 4 | -3 | 328 |
| A1 Activation-only | 139 | 2 | 2 | 0 | 107 |
| A2 Full Opportunity | 139 | 1 | 1 | 0 | 88 |
| Random matched | 140 | 2 | 1 | +1 | 88 |
| Blind matched | 139 | 1 | 1 | 0 | 88 |
| Diagnosis-only | 139 | 0 | 0 | 0 | 0 |

H1（A2 相对 random 的 net rescue）未成立。H2（A2 相对 A1 减少 no-opportunity intervention）成立，Holm-adjusted $p=7.63\times10^{-6}$。A2 消除了 19 次 A1 干预，节省 19 calls 和 9,149 tokens，regressions 从 2 降到 1，同时少获得 1 个 rescue。该结果表明 Full Opportunity 在当前 pipeline 中改善干预选择性和计算分配，但未提高最终 pass@1。

第二轮 Repair 激活 43 次、消耗 26,118 tokens，没有产生额外 rescue 或 regression。后续瓶颈分析将默认最大修复轮数设为 1。

### 5.4 Verifier 与 Repair 瓶颈

V-live 在 164 个初始候选上的 confusion counts 为 TP=23、FP=41、TN=98、FN=2：precision 35.9%，recall 92.0%，specificity 70.5%。对 139 个官方正确候选，false-trigger rate 为 29.5%。41 个 false failures 的离线审计得到：

- 27 个 generated-test false alarm；
- 14 个 formatting/interface mismatch；
- 1 个最终 regression。

从 25 个 baseline failures 出发，23 个被 V-live 检出，18 个同时满足 A2 opportunity 与 actionable evidence，18 个进入首轮 Repair，最终仅 1 个被修复。条件成功率为：

$$
P(\text{rescue}\mid\text{A2 opportunity and actionable evidence})=1/18=5.6\%.
$$

历史 Repair artifacts 保存 normalized code 与 official result，但未保存 pre-normalization raw wrapper。因此，17 个未修复 case 无法进一步可靠拆成 semantic generation、extraction 和 normalization failure；重复 evaluator 未显示 execution instability。

### 5.5 C1 candidate preservation

C1 为 Solver-A/Solver-B→Judge→V-live→conditional Repair。其结果为：

| 指标 | 数值 |
|---|---:|
| 至少一个分支候选正确 | 150/164 |
| Judge 保留正确结果 | 65/150 |
| Aggregation destruction | 85 |
| 最终正确 | 131/164 |
| 相对 initial Coder rescues/regressions | 7/15 |

85 个 destruction 中，76 个来自 Judge synthesis/修改后破坏正确候选，9 个来自直接选择错误候选；26 个 case 同时存在 candidate conflict 或非区分性 evidence。输入可见性完整，保存的 Judge body 可由 extraction 重现。

| Aggregation policy | Correct | Rescues | Regressions | Calls | Tokens |
|---|---:|---:|---:|---:|---:|
| Current C1 | 131 | 7 | 15 | 643 | 282,210 |
| A-simple | 125 | 6 | 20 | 492 | 204,598 |
| Preserve-Then-Repair | 139 | 7 | 7 | 574 | 251,410 |
| Oracle preserve | 150 | 11 | 0 | 492 | 204,598 |

A-simple 依赖 noisy V-live evidence，表现低于 Current C1。Preserve-Then-Repair 将 regressions 从 15 降到 7，并减少 69 calls 与 30,800 tokens，但最终为 139/164，未超过 initial candidate。Oracle preserve 是使用 official correctness 的离线 upper bound，不是可部署策略。

### 5.6 Component upper bounds

| 替换条件 | Final correct | 相对 current 的变化 | 解释 |
|---|---:|---:|---|
| T2 current V-live+A2+R-live | 139 | 0 | 当前 live conditional pipeline |
| T2 oracle verifier | 140 | +1 | 已保存 oracle-gated run |
| C1 current | 131 | 0 | 当前 Judge+Repair |
| C1 oracle verifier | [132,141] | [+1,+10] | 9 个 case 缺少 Repair counterfactual |
| C1 oracle-preserving aggregator | 150 | +19 | 分支中有正确候选时直接保留 |

Aggregation 具有最大的可识别单组件 headroom。Verifier 的 29.5% false-trigger rate 构成上游安全与成本瓶颈；Repair 的 1/18 成功率限制了正确触发后的收益。当前数据把主要瓶颈定位为 Verification/Aggregation，Repair 为次要瓶颈。

### 5.7 Omni-MATH

| Workflow | Accuracy | Rescues/Regressions | Joint failure | $\phi$ | Token ratio |
|---|---:|---:|---:|---:|---:|
| G0 | 57.00% | — | 60.89% | 0.868 | 1.000× |
| G1 Debate | 56.56% | 56/60 vs G0 | 46.56% | 0.554 | 1.018× |
| Adversarial Critic | 58.67% | 63/44 vs G1 | 44.78% | 0.539 | 1.182× G1 |
| G1-Preserve | 56.44% | 60/61 vs G1 | 45.89% | 0.688 | 0.997× G1 |

G1 将 joint failure 降低 14.33 个百分点，并把 one-correct cases 从 55 提高到 204；最终 accuracy 相对 G0 下降 0.44 个百分点（clustered bootstrap CI 跨 0，McNemar $p=0.781$）。Adversarial Critic 相对 G1 增加 2.11 个百分点，但 CI 跨 0（$p=0.081$）。G1-Preserve 将 IndependentSolver correct→wrong 从 67 降到 27，却没有改善最终 accuracy。Omni-MATH 支持 topology intervention 改变 failure structure，不支持稳定性能提升。

### 5.8 MuSiQue

MuSiQue G0 为 130/300 EM=43.33%。Evidence expansion 为 127/300（42.33%），8 rescues、11 regressions；matched random expansion 为 111/300（37.00%），8 rescues、27 regressions。Targeted 相对 random 高 5.33 个百分点，paired bootstrap 95% CI 为 $[+1.67,+9.00]$，McNemar $p=0.0070$；相对 G0 则为 -1.00 个百分点，CI 跨 0。

该结果显示 target selection 减少了相对 random 的损害，但没有恢复 baseline。自然轨迹中的 Evidence weakness 与 119 次 Aggregator blame reduction来自 lexical/contract proxy；语义 gold 尚未完成，因而不作为 semantic localization accuracy。

## 6. 讨论

### 6.1 Diagnosis improvement 为何没有转化为 utility

Full Opportunity 改善了归因和干预选择，却没有改善 HumanEval 最终结果。Failure waterfall 给出具体损失位置：

```text
25 baseline failures
  → 23 detected by V-live
  → 18 judged actionable by A2
  → 18 repaired
  → 1 repair success
  → 1 final rescue

139 baseline-correct candidates
  → 41 false-triggered by V-live
  → 27 entered A2 intervention
  → 1 damaged
  → 1 final regression
```

因此，当前诊断不是主要效用瓶颈。Verifier 提供了高 recall、低 precision 的 trigger；Repair 在有效机会下的成功率较低；C1 又在触发前引入 candidate destruction。局部诊断改善被后续组件的误差抵消。

### 6.2 T1 与 T2/C1 的机制差异

历史 T1 直接处理由 official evaluator 确认的 25 个 Coder failures，向 Coordinator 提供 failure report、Planner specification 和实现结果，并让 139 个正确 candidate 绕过 repair。T2 使用 generated-test verifier 选择 cohort；C1 还在验证前加入 Solver-B 与 Judge synthesis。

T1 的优势与三项设计相关：可靠 failure cohort、较完整 repair evidence、正确候选旁路保留。T2/C1 的额外转换增加了 false trigger、信息破坏和 regression opportunity。该比较解释了简单 Coordinator repair 与复杂 conditional/compositional pipeline 的效用差异。

### 6.3 Opportunity 的当前作用

A2 在 live pipeline 中首先表现为 selective computation：相对 A1 少 19 次干预、9,149 tokens 和 1 个 regression。其定位优势没有转化为更多 rescue。后续优化应分别提高 verifier evidence quality、repair execution 和 candidate preservation，并保持冻结诊断作为独立变量。

### 6.4 结构复杂度与性能

C1 验证了冻结诊断能够在原语组合上运行，也暴露了 aggregation destruction。复杂结构提供更多候选和恢复路径，同时增加选择与改写风险。结构复杂度本身不是收益指标；评价需要同时报告 activated path、candidate availability、preservation、rescues/regressions 与成本。

## 7. 局限性

1. HumanEval 的自然语义 source ground truth 尚未完成；自然执行主要通过配对结果和 operational audit 评估。
2. Compositional actionability label 由冻结 contract/opportunity 字段构造，可用于 origin/actionability 分解，但不是独立人工 gold。
3. 历史 Repair response 缺少 pre-normalization wrapper，限制了 semantic、extraction 与 normalization failure 的细分。
4. Oracle verifier、oracle selection 和 oracle preserve 只用于离线 upper-bound decomposition。
5. HumanEval 的主要结果来自单 backbone 和固定 artifacts；外部模型与 seed 稳健性尚未测量。
6. MuSiQue 的自然 opportunity 仍依赖 lexical/contract proxy；语义盲审尚未完成。
7. 当前 compositional study 覆盖 P1–P4 与 C1；SharedState 和层级 C2 未进入实验。

## 8. 结论

EntroFlow 在激活执行图上按 execution instance 计算机会条件化归因。HumanEval compositional faults 中，Full Opportunity 将注入源定位从 Activation-only 的 53.7% 提高到 82.4%，并消除 inactive 与 no-opportunity blame；冻结规则在 C1 上无需 amendment。自然 HumanEval 执行则显示，定位改善主要减少无依据干预和计算量，未提高最终 pass@1。

D/V/S/R/A 分解将效用损失定位到三个环节：V-live 对官方正确候选产生 41 个 false triggers；Repair 在 18 个有效机会中仅成功 1 次；C1 Judge 破坏 85 个已有正确候选。历史 Coordinator–Worker 的 11 个净 rescues 表明诊断可以支持有效干预，但有效性依赖 trigger reliability、repair evidence 和 candidate preservation。Omni-MATH 与 MuSiQue 的结果同样区分了机制变化、相对 harm reduction 和最终性能。

当前证据支持将多智能体工作流优化建模为：

$$
\boxed{
\text{Diagnose}
\rightarrow \text{Trigger}
\rightarrow \text{Select}
\rightarrow \text{Repair}
\rightarrow \text{Preserve/Aggregate}
\rightarrow \text{Paired Evaluate}
}
$$

下一步实验应在保持诊断冻结的条件下，分别测试 verifier calibration、repair evidence 与 candidate-preserving aggregation。
