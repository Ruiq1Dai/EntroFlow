# EntroFlow

EntroFlow 是一个面向多智能体工作流的框架无关诊断库。它从保存的轨迹重建工作流拓扑，依据节点契约和实际可见输入区分局部失败与传播失败，并用配对结果和成本评估局部工作流修改。

核心包不依赖 AutoGen 或特定模型服务。框架适配器只需记录拓扑、节点可见上下文、输出、provenance 和最终任务结果。

## 主要能力

- Sequence、Fork/Join、条件路由和 bounded feedback 静态原语；
- 带 execution-instance identity 的样本级激活执行图；
- 不依赖 topology-specific rule 的机会条件化节点与边归因；
- 带显式成本约束的局部 rewrite proposal 和 paired comparison；
- 可选的 AutoGen 与 embedding 集成。

## 安装

```bash
python -m pip install -e .
```

开发环境：

```bash
python -m pip install -e '.[dev]'
```

可选集成：

```bash
python -m pip install -e '.[autogen,embedding]'
```

EntroFlow 需要 Python 3.10 或更高版本。

## 快速示例

```python
import numpy as np

from entroflow.entropy import RollingEdgeEntropy

monitor = RollingEdgeEntropy(window_size=32, min_samples=8)
event = monitor.observe("retriever_to_reasoner", np.asarray([0.1, 0.2, 0.3]))

if event.ready:
    print(event.entropy, event.delta)
```

拓扑检查、轨迹字段、诊断契约和局部 rewrite 流程见
[topology optimizer 文档](docs/topology_optimizer_plugin.md)。

## 仓库结构

```text
src/entroflow/   核心库
tests/           单元与集成测试
docs/            长期维护的架构和研究文档
experiments/     可复现实验 runner；生成结果默认忽略
```

仓库不保存大型数据集、checkpoints、模型输出、运行日志或自动生成报告。实验命令会在本地创建输出目录。

## 开发验证

```bash
python -m pytest -q
ruff check .
python -m build
```

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)，许可证见 [LICENSE](LICENSE)。
