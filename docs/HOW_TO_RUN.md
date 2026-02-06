# CD_Circuit HOW_TO_RUN（隔离、可复现、详细版）

本指南基于当前仓库实际内容编写，并严格符合你的服务器使用约束。

## 0. 强制约束（本指南已落实）

1. 所有命令都在 conda 环境 `cdcircuit` 中执行。
2. 不修改服务器全局设置，不影响其他项目。
3. HuggingFace 缓存统一复用 `/mnt/nfs/zijie/huggingface_cache`。
4. 所有输出、日志、图像和权重统一写入 `/mnt/nfs/zijie/cd_circuit_output`。
5. 所有步骤都依据本仓库现有结构和 notebook 实际代码。

## 1. 项目结构总览（当前仓库可运行部分）

- 核心代码：`pyfunctions/`
- 论文相关主实验：`notebooks/`
  - `notebooks/greater_than_task.ipynb`
  - `notebooks/IOI_automated_circuit_discovery.ipynb`
  - `notebooks/IOI_analysis_visualization.ipynb`
  - `notebooks/docstring_task_analysis.ipynb`
  - `notebooks/faithfulness_eval.ipynb`
- greater-than 已包含本地缓存资源：
  - `greater_than_task/cache/potential_nouns.txt`
  - `greater_than_task/cache/logit_indices.pt`
- 额外功能 notebook：
  - `notebooks/Local_importance.ipynb`（有附加依赖与缺失文件，见第 8 节）

## 2. 每次会话启动前设置（不产生全局副作用）

每次新开 shell 后先执行：

```bash
conda activate cdcircuit
cd /raid/home/zijie/projects/Fuze_MI_repro/CD_Circuit

# Required shared paths
export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/transformers
export CD_OUTPUT_DIR=/mnt/nfs/zijie/cd_circuit_output

# Keep temp/cache writes out of global home caches
export XDG_CACHE_HOME=/mnt/nfs/zijie/cd_circuit_output/.cache
export MPLCONFIGDIR=/mnt/nfs/zijie/cd_circuit_output/.mplconfig
export PIP_CACHE_DIR=/mnt/nfs/zijie/cd_circuit_output/.pip_cache

mkdir -p "$CD_OUTPUT_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
```

注意：
- 不要把这些 `export` 写入 `~/.bashrc` 或系统级配置。
- 不要使用 `sudo` 或系统级安装。
- 仅在 `cdcircuit` 环境里安装/变更依赖。

## 3. 依赖安装（仅限 `cdcircuit` 环境）

```bash
conda activate cdcircuit
cd /raid/home/zijie/projects/Fuze_MI_repro/CD_Circuit
pip install -r requirements.txt
```

说明：
- `requirements.txt` 覆盖 CD-T 核心流程。
- 个别可选 notebook 还需要额外依赖（见第 8 节）。

## 4. 推荐输出目录组织

每次实验建议创建独立运行目录：

```bash
export RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_DIR=$CD_OUTPUT_DIR/$RUN_ID
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
```

建议在 `$RUN_DIR` 下按类型存放：
- `$RUN_DIR/figures`
- `$RUN_DIR/results`
- `$RUN_DIR/checkpoints`
- `$RUN_DIR/logs`

## 5. 安全启动 Jupyter

这些 notebook 使用了：

```python
base_dir = os.path.split(os.getcwd())[0]
sys.path.append(base_dir)
```

因此需要从 `notebooks/` 目录启动 Jupyter：

```bash
conda activate cdcircuit
cd /raid/home/zijie/projects/Fuze_MI_repro/CD_Circuit/notebooks
jupyter lab --no-browser --ip=127.0.0.1 --port=8890
```

这样可以保证 `from pyfunctions...` 的导入路径正确。

## 6. 主要实验运行说明（论文相关）

## 6.1 Greater-than 自动电路发现

Notebook：`notebooks/greater_than_task.ipynb`

功能：
- 用 TransformerLens 加载 `gpt2-small`。
- 从 `../greater_than_task/cache/` 使用本地资源构造 `YearDataset`。
- 执行 CD-T 分解与消融分析。

首次运行行为：
- 如果 `/mnt/nfs/zijie/huggingface_cache` 中已有 `gpt2-small`，将直接复用。
- 若不存在，会下载到该缓存目录。

建议增加结果保存单元（放到结果相关单元后）：

```python
import os, json, torch
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
(run_dir / "results").mkdir(parents=True, exist_ok=True)

torch.save(
    {"results": results},
    run_dir / "results" / "greater_than_results.pt"
)
```

## 6.2 IOI 自动电路发现

Notebook：`notebooks/IOI_automated_circuit_discovery.ipynb`

功能：
- 加载 `gpt2-small`。
- 构建 `IOIDataset` 并进行 head 级分解。
- 迭代筛选关键节点并评估效果。

建议保存方式：

```python
import os, json
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
(run_dir / "results").mkdir(parents=True, exist_ok=True)

with open(run_dir / "results" / "ioi_top_heads.json", "w") as f:
    json.dump([(str(r.ablation_set), float(r.score)) for r in results[:200]], f)
```

## 6.3 IOI 分析与可视化

Notebook：`notebooks/IOI_analysis_visualization.ipynb`

功能：
- 与 IOI 分析流程类似，重点是结果分析与可视化。

在任何 `plt.savefig(...)` 之前先确保：

```python
from pathlib import Path
import os
fig_dir = Path(os.environ["RUN_DIR"]) / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(fig_dir / "some_plot.png", dpi=300, bbox_inches="tight")
```

## 6.4 Docstring 任务分析

Notebook：`notebooks/docstring_task_analysis.ipynb`

功能：
- 使用 `attn-only-4l` 模型和 `im_utils/` 的 prompt 工具。
- 在 docstring 风格任务上执行 CD-T 分解与电路分析。

同样遵守：
- 模型缓存走 `/mnt/nfs/zijie/huggingface_cache`
- 输出保存到 `$RUN_DIR/...`

## 6.5 Faithfulness 评估

Notebook：`notebooks/faithfulness_eval.ipynb`

重要说明：
- 该 notebook 内存在 `REDACTED` 形式的占位路径。
- 需要替换为 `$RUN_DIR` 下真实路径，例如：
  - `$RUN_DIR/results/rand_faith_scores.json`
  - `$RUN_DIR/results/cdt_faith_scores.json`
  - `$RUN_DIR/figures/cdt-vs-random.pdf`

## 7. HuggingFace 缓存复用检查

检查当前 shell 是否正确指向共享缓存：

```bash
echo "$HF_HOME"
echo "$HUGGINGFACE_HUB_CACHE"
echo "$TRANSFORMERS_CACHE"
```

三者都应指向 `/mnt/nfs/zijie/huggingface_cache...`。

实际行为：
- 缓存中已存在的模型会直接复用，不重复下载。
- 不存在时只下载一次，后续运行继续复用。

## 8. 已知缺口 / 非核心流程补充

## 8.1 `notebooks/Local_importance.ipynb`

该 notebook 在当前仓库状态下不是开箱即用：
- notebook 单元里会安装额外包（`lime`, `shap`, `captum`）。
- `pyfunctions/local_importance.py` 依赖：
  - `pyfunctions.config`（当前仓库缺失）
  - `pyfunctions._integrated_gradients`（当前仓库缺失）
- 其中引用的 pathology 模型/数据路径也未在仓库提供。

若你需要运行该 notebook：
- 仅在 `cdcircuit` 环境安装额外依赖：
  - `pip install lime shap captum`
- 缺失文件请只在当前项目目录补齐，不做全局修改。

## 8.2 Pathology/BERT 私有资产

仓库 README 已说明：部分 pathology 数据与模型未公开上传。
因此 pathology 相关复现实验需要你提供兼容的数据与模型资产。

## 9. 最小可复现流程（建议首次运行）

1. 执行第 2 节会话环境设置。
2. 执行第 3 节依赖安装。
3. 按第 5 节方式启动 Jupyter。
4. 优先跑 `notebooks/greater_than_task.ipynb`。
5. 输出统一保存到 `$RUN_DIR/results` 和 `$RUN_DIR/figures`。
6. 再运行 IOI 相关 notebook。

## 10. 快速排障

- `ModuleNotFoundError: pyfunctions...`：
  - 通常是 Jupyter 启动目录不对。
  - 需从 `.../CD_Circuit/notebooks` 启动。
- CUDA OOM：
  - 调小 notebook 内批量参数（如 `NUM_SAMPLES`, `NUM_AT_TIME`）。
- 缓存路径不正确：
  - 回到第 2 节重新检查并设置当前 shell 的环境变量。
- 输出写到了项目目录零散位置：
  - 将所有保存逻辑统一改为 `Path(os.environ["RUN_DIR"]) / ...`。

---

本指南刻意避免全局配置改动，并确保模型缓存与实验输出都落在你要求的 NFS 路径下。

## 11. 论文算法与项目功能详解（CD-T 核心原理）

这一部分解释“为什么这些 notebook 能做电路发现”，以及项目代码如何把论文思路变成可执行流程。

### 11.1 CD-T 的目标

CD-T（Contextual Decomposition for Transformers）的目标是：
- 给定一个任务目标（例如 IOI 的 logit diff，或 greater-than 的正确年份预测），
- 定量评估每个中间组件（通常是某层某位置某 attention head）对该目标的贡献，
- 进而自动筛出“电路”（重要组件集合），再用消融验证其 faithful 程度。

### 11.2 关键表示：`rel / irrel` 分解

几乎所有核心函数都围绕这一点：
- 把每个中间激活写成 `total = rel + irrel`
- `rel` 表示“与指定 source node 或 source token 相关”的部分
- `irrel` 表示其余部分

实现上，线性层、LayerNorm、激活函数、Attention 都分别定义了分解传播规则，并尽量保持：
- 可加性：`rel + irrel` 与原模型前向一致
- 数值稳定性：用 `normalize_rel_irrel` 处理符号冲突与漂移

### 11.3 Transformer 中如何做分解传播

在 attention 子层中，代码会显式处理：
- Q/K 线性投影的 rel/irrel 分配
- 注意力分数与 softmax 概率的分解近似
- V 投影与 attention 加权求和后的分解
- 头输出拼接后再经过 O 矩阵映射回残差流

在 MLP 子层中，代码会处理：
- `W_in`、激活、`W_out` 的分解传播
- 与残差相加后的归一化

GPT 路径与 BERT 路径的差异点主要在：
- LayerNorm 放置位置（pre/post）
- 模块组织形式（TransformerLens vs HuggingFace BERT）

### 11.4 Source node / Target node 的意义

- Source node：你想问“它贡献了多少”的组件。
- Target node：你想把贡献投射到哪里（可选）。
  - 若 target 为空，通常看对最终 logits 的贡献。
  - 若 target 非空，可看 source 对中间节点的贡献。

对应实现：
- `ablation_set`：一个 source 节点集合（可单节点，也可多节点）
- `TargetNodeDecompositionList`：记录 source->target 的 rel/irrel 结果

### 11.5 为什么有“均值消融（mean ablation）”

仅做打分不够，需要验证“电路是否真能支撑任务”。
本仓库的 faithfulness 验证思路是：
- 保留候选电路节点的输出
- 其余节点用 reference 分布（常见是 ABC 数据集的均值激活）替代
- 看任务指标（如 logit diff）是否仍接近原模型

如果少量节点就能保持高性能，说明电路发现更可信。

### 11.6 与论文复现的对应关系（本仓库）

- greater-than：`notebooks/greater_than_task.ipynb`
- IOI 自动发现：`notebooks/IOI_automated_circuit_discovery.ipynb`
- IOI 可视化分析：`notebooks/IOI_analysis_visualization.ipynb`
- docstring 任务：`notebooks/docstring_task_analysis.ipynb`
- faithfulness 曲线：`notebooks/faithfulness_eval.ipynb`

## 12. 运行功能深解：每类实验到底在做什么

### 12.1 Greater-than 流程

核心步骤：
1. 加载 `gpt2-small`
2. 生成/读取年份比较数据（`YearDataset`）
3. 枚举 source nodes（层、位置、头）
4. 批量计算 source 对目标（通常 logits）的分解分数
5. 取 top 节点形成候选电路
6. 用均值消融做 faithful 验证

结果类型：
- 每个 head 的重要性分数
- 候选电路列表
- 消融后的任务性能变化

### 12.2 IOI 自动电路发现

核心步骤：
1. 构造 IOI 数据与对应 ABC reference 数据
2. 计算用于 patch/ablation 的 head 输出均值
3. 跑 source->logit 分解并排名
4. 迭代生成 target（上一轮 top 节点）并继续 source->target 分解
5. 对候选电路做信度验证

结果类型：
- 节点排序
- 分层迭代筛选轨迹
- faithfulness 或 logit diff 对比曲线

### 12.3 Docstring 任务

核心步骤：
1. 加载 `attn-only-4l`
2. 用 `im_utils/prompts.py` 生成 prompt 对
3. 在 toy model 路径上执行 CD-T 分解
4. 分析被识别出的关键节点

特征：
- 模型无 MLP（attention-only），更利于机制分析
- 更偏“机制可解释性案例”而不是标准 benchmark

### 12.4 Faithfulness 评估

核心步骤：
1. 定义 clean / null 基线（logit diff）
2. 随机电路与 CD-T 电路对比
3. 随节点数增长绘制 faithfulness 曲线

常用指标：
- `logits_to_ave_logit_diff_2`
- `faithfulness = (circuit_score - model_null) / (model_clean - model_null)`

## 13. 逐目录逐文件讲解（代码思路与功能）

本节按目录给出“文件职责 + 使用场景 + 关键函数”。

### 13.1 根目录文件

- `README.md`
  - 项目总览、目录说明、最基础依赖安装说明。
- `HOW_TO_RUN.md`
  - 本运行手册，面向你当前服务器约束定制。
- `requirements.txt`
  - 核心 Python 依赖列表（CD-T 主流程）。
- `Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition.pdf`
  - 论文原文，用于理论与实验对照。
- `package.json` / `package-lock.json`
  - 前端/Node 相关元数据，不是主实验路径必需。

### 13.2 `pyfunctions/`（算法核心）

- `pyfunctions/cdt_core.py`
  - CD 基础算子库（线性层、激活、LayerNorm、attention mask、BERT embedding 等）。
  - 关键思路：定义可复用分解传播原语。
  - 关键函数：
    - `normalize_rel_irrel`
    - `prop_linear_core` / `prop_linear`
    - `prop_act`
    - `prop_layer_norm`
    - `get_extended_attention_mask`

- `pyfunctions/cdt_basic.py`
  - BERT 版本“基础 CD”示例，不含 source patching 的复杂逻辑。
  - 适合理解从输入 token 到分类 logit 的分解链路。

- `pyfunctions/cdt_ablations.py`
  - 通用 patch/ablation 相关函数。
  - 负责在指定 source node 处重写 `rel/irrel`，并支持均值替换。
  - 关键函数：
    - `set_rel_at_source_nodes`
    - `calculate_contributions`
    - `reshape_separate_attention_heads`

- `pyfunctions/cdt_from_source_nodes.py`
  - 从 source node 出发的中间实现（偏 BERT 示例向）。
  - 展示如何在层内 patch 后继续传播并观测贡献。

- `pyfunctions/cdt_source_to_target.py`
  - 项目最关键的“通用 CD-T 主干”。
  - 统一 GPT/BERT 的 source->target / source->logit 分解流程。
  - 含批处理、缓存前层激活、分解得分计算等核心逻辑。

- `pyfunctions/wrappers.py`
  - 适配层：把 TransformerLens GPT 模块包装成与 BERT 路径兼容的访问接口。
  - 定义节点与结果类型：
    - `Node`
    - `AblationSet`
    - `OutputDecomposition`
    - `TargetNodeDecompositionList`

- `pyfunctions/toy_model.py`
  - 针对 `attn-only-4l` 的 CD-T 实现（docstring 实验用）。
  - 重点是“无 MLP”结构下的传播和输出分解。

- `pyfunctions/faithfulness_ablations.py`
  - 来自 IOI/ARENA 体系的 mean ablation hook 工具。
  - 负责构建“保留电路 + 非电路均值替换”的钩子机制。
  - 常用函数：
    - `add_mean_ablation_hook`
    - `compute_means_by_template`
    - `logits_to_ave_logit_diff_2`

- `pyfunctions/ioi_dataset.py`
  - IOI 数据构造与模板组织工具（含 token 对齐、索引信息）。
  - IOI notebook 的数据入口。

- `pyfunctions/general.py`
  - 通用工具函数集合（I/O、列表处理、简单统计等）。
  - 非算法核心，但被多个模块复用。

- `pyfunctions/pathology.py`
  - pathology 任务数据清洗与标签修正工具（私有资产场景）。
  - 与公开论文主实验关联较弱，更多是扩展用途。

- `pyfunctions/local_importance.py`
  - 统一封装 CDT/LIME/SHAP/LIG 的局部解释入口。
  - 当前仓库缺少其依赖的若干文件（见第 8 节）。

- `pyfunctions/__init__.py`
  - 包初始化文件。

### 13.3 `greater_than_task/`

- `greater_than_task/greater_than_dataset.py`
  - 年份比较任务数据集构造（正样本/坏样本、掩码、token）。
  - 定义 `YearDataset`。

- `greater_than_task/utils.py`
  - `get_valid_years`：筛选 tokenizer 下可分成目标 token 结构的年份。

- `greater_than_task/README.md`
  - 说明该目录内容来源（外部仓库复用）。

- `greater_than_task/cache/`
  - `potential_nouns.txt`：句子模板名词表。
  - `logit_indices.pt`：年份 token 索引缓存。
  - `valid_years.pt`、`gelu_12_tied.circ`：任务相关缓存/模型资产。

### 13.4 `im_utils/`

- `im_utils/prompts.py`
  - docstring 任务 prompt 生成器。
  - 定义 `Prompt` 数据结构和多种 corruption 方式。

- `im_utils/variables.py`
  - prompt 采样词表（变量名、常见单词等）。

- `im_utils/README.md`
  - 说明该目录来源于外部 MI 工具项目。

- `im_utils/LICENSE`
  - 许可证文件。

### 13.5 `methods/bag_of_ngrams/`

- `methods/bag_of_ngrams/processing.py`
  - 文本清洗与 n-gram 特征预处理工具（偏 pathology/传统特征路径）。

- `methods/bag_of_ngrams/__init__.py`
  - 包初始化文件。

### 13.6 `notebooks/`（实验入口）

- `notebooks/README.md`
  - notebook 目录总说明。

- `notebooks/greater_than_task.ipynb`
  - greater-than 复现主入口（推荐第一个跑）。

- `notebooks/IOI_automated_circuit_discovery.ipynb`
  - IOI 自动电路发现主流程。

- `notebooks/IOI_analysis_visualization.ipynb`
  - IOI 分析与可视化。

- `notebooks/docstring_task_analysis.ipynb`
  - docstring 任务分析（attn-only-4l）。

- `notebooks/faithfulness_eval.ipynb`
  - faithfulness 曲线与对比评估。

- `notebooks/Local_importance.ipynb`
  - 局部解释方法对比（当前需补额外依赖与缺失模块）。

- `notebooks/correctness_tests/*.ipynb`
  - CD 实现正确性与 sanity check 相关 notebook。

## 14. 追加注意事项（继续强调，避免踩坑）

### 14.1 环境与隔离

- 每次运行前都重新执行第 2 节环境变量导出，不要假设 shell 继承。
- 不要把任何配置写入全局 profile（例如 `~/.bashrc`）。
- 不在其他 conda 环境里混跑 notebook。

### 14.2 缓存与输出

- 任何模型加载前先确认：
  - `HF_HOME`
  - `HUGGINGFACE_HUB_CACHE`
  - `TRANSFORMERS_CACHE`
- 任何 `torch.save` / `plt.savefig` / `json.dump` 都要落到 `$RUN_DIR` 下。
- 避免 notebook 默认把文件写到当前目录，必要时统一封装 `run_dir = Path(os.environ["RUN_DIR"])`。

### 14.3 路径与启动

- 这套 notebook 依赖“从 `notebooks/` 启动 Jupyter”的相对路径假设。
- 若从仓库根目录启动，`sys.path.append(base_dir)` 会偏移，导致导入或文件路径错误。

### 14.4 计算资源

- 先用小批量参数试跑（如 `NUM_SAMPLES`、`NUM_AT_TIME`），确认流程通再扩大规模。
- OOM 优先降 batch，不要先改算法逻辑。
- 若多实验并行，确保每个实验使用独立 `RUN_DIR`，避免结果覆盖。

### 14.5 非开箱模块

- `Local_importance` 与 pathology 相关流程是“扩展模块”，不是本仓库当前主复现路径。
- 缺失文件补齐时，仅在本项目目录内操作，不做系统级安装与全局改动。

### 14.6 结果可追溯

- 建议每个 `RUN_DIR` 保存：
  - 参数快照（json）
  - 关键结果（pt/json）
  - 图像（png/pdf）
  - 简短日志（txt/md）
- 这样可保证不同实验批次之间可比较、可复现、可回滚到具体设置。

## 15. 按阅读顺序的源码导读路径（从入门到可改代码）

这一节给你一条“最少绕路”的阅读路线。每一步都包含：
- 先看哪些文件
- 重点看哪些函数/结构
- 看完你能做什么

### 15.1 第 0 步：先建立全局地图（10-20 分钟）

先看文件：
- `README.md`
- `notebooks/README.md`
- 本文档的第 1、6、11、12、13 节

重点关注：
- 项目主实验入口是 `notebooks/*.ipynb`
- 算法核心在 `pyfunctions/`
- greater-than / IOI / docstring 三条主要任务线

看完你能做什么：
- 知道“从哪个 notebook 进、核心代码在哪、哪些模块是扩展模块”。

### 15.2 第 1 步：理解核心数据结构（15-30 分钟）

先看文件：
- `pyfunctions/wrappers.py`

重点看：
- `Node`
- `AblationSet`
- `OutputDecomposition`
- `TargetNodeDecompositionList`
- `GPTAttentionWrapper`、`GPTLayerNormWrapper`

看完你能做什么：
- 理解 CD-T 在代码里如何表示“一个头/一个节点/一个分解结果”。
- 看懂后续函数签名里的 `ablation_list`、`target_nodes`、`target_decomps`。

### 15.3 第 2 步：读 CD 基础算子（30-60 分钟）

先看文件：
- `pyfunctions/cdt_core.py`

重点看函数（按顺序）：
- `normalize_rel_irrel`
- `prop_linear_core` / `prop_linear`
- `prop_act`
- `prop_layer_norm`
- `mul_att`
- `get_extended_attention_mask`

看完你能做什么：
- 明白 `rel/irrel` 如何在单层算子内传播。
- 能判断一个新算子要如何接入 CD 分解逻辑。

### 15.4 第 3 步：看最简 BERT 路径（30-60 分钟）

先看文件：
- `pyfunctions/cdt_basic.py`

重点看函数（按调用链）：
- `prop_self_attention`
- `prop_attention`
- `prop_layer`
- `prop_encoder`
- `prop_classifier_model_from_level`
- `comp_cd_scores_level_skip`

看完你能做什么：
- 从“输入 token 到分类分数”的完整分解链路能走通。
- 这是理解复杂版本（source->target）的最好跳板。

### 15.5 第 4 步：看 patch/ablation 机制（30-60 分钟）

先看文件：
- `pyfunctions/cdt_ablations.py`
- `pyfunctions/cdt_from_source_nodes.py`

重点看：
- `set_rel_at_source_nodes`
- `calculate_contributions`
- `prop_attention_patched`
- `prop_layer_patched`

看完你能做什么：
- 明白 source node 是怎么被“设为相关”的。
- 明白 target 贡献是如何按层收集的。

### 15.6 第 5 步：主干算法（必读，60-120 分钟）

先看文件：
- `pyfunctions/cdt_source_to_target.py`

重点看函数（建议顺序）：
- `prop_attention_no_output_hh`
- `prop_GPT_layer`
- `prop_BERT_hh`
- `prop_GPT`（主入口）
- `batch_run`（批量运行入口）
- 与 score 相关的辅助函数（文件内的结果计算函数）

看完你能做什么：
- 能完整解释“source -> target / logits”的自动分解流程。
- 能改动实验设定（source 枚举、target 选择、缓存策略、批处理大小）。

### 15.7 第 6 步：先跑通一个任务（推荐 greater-than）

先看文件：
- `greater_than_task/greater_than_dataset.py`
- `greater_than_task/utils.py`
- `notebooks/greater_than_task.ipynb`

重点看：
- `YearDataset`
- `get_valid_years`
- notebook 中 source node 枚举和 `prop_GPT` / `batch_run` 调用

看完你能做什么：
- 能独立复现实验并保存结果到 `$RUN_DIR`。
- 能修改样本规模、评分方式、候选电路筛选策略。

### 15.8 第 7 步：读 IOI 自动发现与 faithfulness（60-120 分钟）

先看文件：
- `pyfunctions/ioi_dataset.py`
- `pyfunctions/faithfulness_ablations.py`
- `notebooks/IOI_automated_circuit_discovery.ipynb`
- `notebooks/faithfulness_eval.ipynb`

重点看：
- `IOIDataset` 的构造与 token 索引
- `add_mean_ablation_hook`
- `compute_means_by_template`
- `logits_to_ave_logit_diff_2`

看完你能做什么：
- 能解释“为什么需要 ABC 均值参考”。
- 能复现并比较 random circuit 与 CD-T circuit 的 faithfulness 曲线。

### 15.9 第 8 步：读 toy model / docstring 路径（45-90 分钟）

先看文件：
- `im_utils/prompts.py`
- `im_utils/variables.py`
- `pyfunctions/toy_model.py`
- `notebooks/docstring_task_analysis.ipynb`

重点看：
- `Prompt` 结构
- prompt 生成函数（docstring 系列）
- `prop_toy_model_4l` / `prop_toy_model_4l_layer`

看完你能做什么：
- 能在 attention-only 架构上复用 CD-T，并自定义 prompt 机制分析任务。

### 15.10 第 9 步：扩展模块按需阅读

按需看文件：
- `pyfunctions/local_importance.py`
- `pyfunctions/pathology.py`
- `methods/bag_of_ngrams/processing.py`

重点说明：
- 这条线不是论文主复现路径。
- 当前仓库缺少 `local_importance` 依赖的部分文件，先不要把它当主线。

看完你能做什么：
- 评估是否值得在你本地补齐缺失模块后再做方法对比实验（LIME/SHAP/LIG/CDT）。

## 16. 阅读完成后的实战里程碑

如果你按第 15 节读完，建议用下面 4 个里程碑检查是否真正掌握：

1. 你能清楚说出 `Node`、`AblationSet`、`OutputDecomposition` 分别对应什么。
2. 你能在 `notebooks/greater_than_task.ipynb` 里改一套 source 枚举策略并跑出新结果。
3. 你能在 IOI 路径上解释并修改 mean ablation 的 reference 生成方式。
4. 你能新增一个保存单元，把关键中间结果稳定输出到 `$RUN_DIR/results`。
