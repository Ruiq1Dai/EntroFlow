# EntroFlow 当前 Entropy-Repair 机制

本文档记录当前代码中的实现，供讨论和后续方法优化使用。它描述的是现状，不是最终方法定义。

## 1. 观测对象

EntroFlow 不直接读取答案或 gold label，只观察 AutoGen collaboration trace。每次工作流决策前，路由器读取最近一条角色消息，并根据静态拓扑确定下一角色：

```text
Planner -> Evidence -> Reasoner -> Aggregator
                         ^
                         |
                    Validator（修复时插入）
```

对每条有向通信边建立独立的 rolling history，例如 `evidence_to_reasoner`。当前实现把消息文本送入固定 embedding 模型，得到一个向量。

## 2. Rolling semantic entropy

实现位于 `src/entroflow/entropy.py` 的 `RollingEdgeEntropy`。

对每条边维护一个固定长度的 embedding 队列：

- `window_size = 32`：最多保留最近 32 条消息；
- `min_samples = 8`：少于 8 条时不计算 entropy；
- `max_clusters = 4`：最多聚成 4 个语义簇；
- 使用 KMeans，`random_state = 42`。

当窗口达到最小样本数后：

1. 对窗口内 embedding 做 KMeans；
2. 统计各 cluster 的比例 `p_i`；
3. 计算 Shannon entropy：

   `H = - sum_i p_i log(p_i)`

4. 用当前 cluster 数归一化到 `[0, 1]`：

   `H_norm = H / log(number_of_clusters)`

5. 保存该边上一次的 entropy，并计算变化量：

   `delta = H_norm(t) - H_norm(t-1)`

因此当前方法同时使用两个信号：

- `entropy`：当前通信窗口的语义多样性/不确定性；
- `delta`：通信不确定性相对前一时刻的上升或下降。

需要注意：当前 entropy 是基于 embedding 聚类分布的 semantic dispersion，不是 token-level predictive entropy，也不是模型 logits 的 uncertainty。

## 3. 是否触发干预

实现位于 `src/entroflow/routing.py` 的 `EntropyPolicy`。

对当前边计算：

```text
anomalous = event.ready
            and delta is not None
            and abs(delta) >= entropy_threshold
```

默认 `entropy_threshold = 0.15`。在满足以下任一条件前不干预：

- 窗口样本数不足 `min_samples`；
- 没有上一时刻 entropy，无法计算 delta；
- `abs(delta)` 小于阈值。

因此当前策略主要是 **变化检测**，不是单纯的高 entropy 检测。一个窗口即使 entropy 很高，只要没有明显变化，也不会触发 repair。

## 4. 如何选择 repair action

当前策略先根据固定拓扑得到 baseline next role，然后仅在 anomaly 条件下根据消息角色、`delta` 方向和变化幅度选择动作。

严重程度定义为：

```text
severe = abs(delta) >= 2 * entropy_threshold
```

当前映射如下：

| 当前消息角色 | entropy 变化 | 当前动作 | 下一角色 | 设计直觉 |
|---|---:|---|---|---|
| Evidence | `delta > 0` 且 severe | `INSERT_VALIDATION` | Validator | 语义分布突然发散，先验证证据 |
| Evidence | `delta > 0` 且非 severe | `CROSS_CHECK` | Validator | 有分歧但程度较低，交叉核验 |
| Evidence | `delta < 0` | `REORDER_VALIDATION` | Validator | 熵回落但轨迹仍异常，提前重排验证 |
| Reasoner | anomaly 且 severe | `EVIDENCE_BYPASS` | Aggregator | 推理异常严重，允许聚合器直接看证据 |
| Reasoner | anomaly 且非 severe | `COMPRESS_CONTEXT` | Aggregator | 上下文可能过于发散，压缩可见上下文 |
| 其他角色 | anomaly | baseline action | baseline next role | 当前没有角色特定 repair |

原有动作仍保留：

- `RETRY_WITH_CONSTRAINTS`
- `INSERT_VALIDATION`
- `EVIDENCE_BYPASS`

新增动作：

- `REORDER_VALIDATION`
- `EXPAND_EVIDENCE`
- `CROSS_CHECK`
- `COMPRESS_CONTEXT`

其中 `EXPAND_EVIDENCE` 已加入动作库，但当前 `EntropyPolicy` 的主分支尚未选择它；这是后续需要明确语义并接入的待优化点。

## 5. repair 如何进入模型上下文

路由器把动作写入：

- `RoleRequest.metadata["repair_action"]`；
- trace event 的 `repair_action`；
- 发给角色的上下文文本：`Runtime repair action: <action>`。

动作还会改变下一角色和 `visible_roles`，从而改变该角色能看到的历史消息。例如 evidence anomaly 触发 Validator 时，Validator 看到 Planner/Evidence 历史；reasoner anomaly 触发 bypass 时，Aggregator 看到 Evidence/Reasoner 历史。

当前版本还会把 repair 具体化为可审计的 recommendation，而不再只写入
动作枚举值。每次异常至少记录：

- `diagnosis`：熵信号对应的可证伪错误假设；
- `repair_target`：要重新检查的链路或约束；
- `repair_instruction`：接收角色必须执行的检查动作；
- `confidence`：仅表示路由器对该假设的信号置信度，不是错误概率。

这些字段同时进入请求上下文和 trace metadata。后续审计应检查接收角色是否
执行了 instruction，并以 paired rescue/regression 评估修复是否有效；不能把
出现 `repair_action` 标签当作修复成功。

## 6. 当前隐含假设

1. embedding 空间中的 cluster 分散度能够反映通信不确定性；
2. entropy 的突变比绝对 entropy 更接近错误发生时刻；
3. `delta` 的正负可以区分“发散”和“收敛后的异常”；
4. 消息发送角色是选择 repair 类型的重要上下文；
5. 一次 repair 的局部干预足以改变后续工作流；
6. 固定阈值和固定窗口在不同 hop 数、不同角色边上具有可迁移性；
7. repair action 通过改变可见历史和下一角色，能够产生实际行为差异，而不只是 trace 标签变化。

## 7. 与导师讨论时最值得优化的点

## 8. 当前新增的局部薄弱拓扑信号

仅使用最高 semantic entropy 会把“多个合理候选”与“约束传递失败”混在一起。
当前路由器增加了一个不依赖 gold 的 handoff-contract signal：

```text
contract_score = max(
    structured_status_score,
    unresolved_marker_score,
    evidence_reference_score
)
```

其中 `structured_status_score` 来自 `insufficient/uncertain/invalid/unparsed`，
`unresolved_marker_score` 检查响应是否明确报告 `missing/unsupported/not
explicitly/contradiction` 等未解决跳点，`evidence_reference_score` 只在前两类
信号已经出现且缺少引用时生效。分数超过阈值时，路由器优先选择当前边的针对性
repair：Evidence 缺桥接关系时 `EXPAND_EVIDENCE`，Reasoner 出现不一致时
`INSERT_VALIDATION`；熵只作为严重程度和变化检测的辅助信号。

在 20 条 pilot 的独立审计上，按角色顺序选择第一个契约违例点得到 16/20
（80%）命中率；该结果是探索性结果，不能替代 held-out 校准。它支持的研究
假设也更窄、更可检验：熵用于发现通信不稳定性，契约违例用于确定局部薄弱边，
repair action 则针对该边丢失的约束类型。

### 7.1 Entropy 定义

- cluster 数是否应固定，而不是随窗口大小和 unique vector 数变化；
- 使用 `log(max_clusters)` 还是当前 cluster 数做归一化；
- rolling window 是否应按角色/hop 使用不同大小；
- 是否加入 absolute entropy、delta、二阶变化 `delta(t)-delta(t-1)`；
- 是否改用 conditional entropy、transition surprisal 或 source-target alignment；
- KMeans 的随机性、embedding 模型和距离度量是否影响结论。

### 7.2 触发条件

- 固定阈值是否应改为 edge-specific baseline / z-score；
- 是否同时要求高 absolute entropy 和高 positive delta；
- 是否需要连续两个异常点才触发；
- 是否按 hop、角色或边分别校准阈值；
- anomaly 是否需要结合 response status、evidence coverage 等非 gold runtime signals。

### 7.3 动作选择

- 当前规则是手工 decision table，是否改为可解释的 cost-sensitive policy；
- 每种 action 的上下文差异是否真正实现，而不是只记录 action 名称；
- `EXPAND_EVIDENCE` 应具体增加哪些证据请求或检索范围；
- `CROSS_CHECK` 是否应调用独立 Validator、重复 Evidence，还是比较两组证据；
- `COMPRESS_CONTEXT` 应采用摘要、筛选 evidence，还是限制角色可见历史；
- 是否允许根据 repair 后 entropy 是否回落决定继续、切换或回退。

### 7.4 评估设计

- 所有 controller 必须使用相同动作库、相同预算和逐题配对样本；
- 记录触发率、动作分布、额外 token、修复成功率和 regression rate；
- 分别评估 2/3/4-hop 和 retrieval dead-end；
- 通过 action ablation 区分“entropy 判断是否干预”和“entropy 选择哪种动作”的贡献。
