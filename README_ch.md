# EntroFlow

EntroFlow 是面向多智能体工作流的运行时 plug-in。它观察 collaboration traces，计算滚动的边级 communication entropy，并在受限动作库中选择通信链接、执行顺序或验证路径的修复动作。

论文比较固定 backbone、固定 MuSiQue 划分、固定 AutoGen 初始工作流和固定推理预算下的系统表现。EntroFlow 不是新的 QA backbone，也不替代 AutoGen。

当前研究协议见 [docs/research_scope.md](docs/research_scope.md)。历史 MA-Base/V4 代码、轨迹、结果和文档已经移除，不能再作为论文证据。

```bash
pip install -e '.[dev]'
pip install 'autogen-agentchat>=0.4' 'autogen-core>=0.4'
```

核心包只提供框架无关的边级熵监测；AutoGen 负责 agent 执行与消息传递，EntroFlow 负责读取 trace 并提出受限 repair action。

## 模型配置

密钥放在仓库根目录的 `.env`，不要写入源码或提交到 Git：

```bash
cp .env.example .env
```

```text
MODEL_API_KEY=你的密钥
MODEL_BASE_URL=服务商提供的 OpenAI-compatible /v1 地址
MODEL_NAME=deepseek-v4-flash
```

AutoGen 不会自动读取该文件。运行实验前加载环境变量：

```bash
set -a
source .env
set +a
```

## MuSiQue baseline

先运行不含 repair 的静态 AutoGen workflow，避免把后续 EntroFlow 增益混入基线：

```bash
python experiments/musique/run.py \
  --data data/MuSiQue/musique_ans_v1.0_dev.jsonl \
  --output artifacts/static_dev/results.jsonl \
  --traces artifacts/static_dev/traces.jsonl \
  --summary artifacts/static_dev/summary.json \
  --failures artifacts/static_dev/failures.jsonl \
  --policy static \
  --target-failures 200 \
  --resume
```

`summary.json` 中的 `accuracy` 是 normalized exact match 的均值；`macro_f1` 是逐题 token F1
的宏平均。15.9 accuracy 与 39.85/44.05 F1 属于不同指标，不能直接比较。失败样本及其完整
AutoGen 通信事件按 `run_id` 写入 `failures.jsonl`，作为后续轨迹熵分析的输入。

修复策略必须使用相同 `example_id` 做逐题配对，报告救回（baseline 错、repair 对）、回归
（baseline 对、repair 错）和额外 token，而不能只比较总体 accuracy：

```bash
python experiments/musique/evaluate_paired_repairs.py \
  --baseline artifacts/static_stratified_20/results.jsonl \
  --candidate artifacts/random_repair_20/results.jsonl artifacts/entropy_repair_20/results.jsonl \
  --output artifacts/paired_repair_evaluation.json
```

按 MuSiQue dev 的 hop 分布收集 200 条失败轨迹（2-hop/3-hop/4-hop 为 104/63/33）：

```bash
python experiments/musique/run.py \
  --data data/MuSiQue/musique_ans_v1.0_dev.jsonl \
  --output artifacts/static_failures_200/results.jsonl \
  --traces artifacts/static_failures_200/traces.jsonl \
  --summary artifacts/static_failures_200/summary.json \
  --failures artifacts/static_failures_200/failures.jsonl \
  --policy static --seed 42 --target-failures-by-hop 104 63 33 --concurrency 6 --resume
```
