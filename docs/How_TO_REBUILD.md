# CD_Circuit 从零重构实施手册（How_TO_REBUILD）

本文档用于指导你在新目录 `~/projects/Fuze_MI_repro/CD_Circuit-repro` 从零重构 CD-Circuit 项目，目标是：
- 完全去 notebook 化（以 `.py` 模块和 CLI 脚本为主）
- 保留论文核心算法与实验功能（greater-than / IOI / docstring / faithfulness）
- 采用可维护、可测试、可复现实验工程结构
- 严格遵守你的环境约束与服务器隔离要求

---

## 0. 约束与目标（必须先确认）

### 0.1 强制约束

1. Conda 环境继续使用 `cdcircuit`。
2. 依赖基于原项目 `requirements.txt`（可在此基础上增补测试和 CLI 工具）。
3. HuggingFace 缓存统一使用：`/mnt/nfs/zijie/huggingface_cache`。
4. 所有输出（日志、图像、权重、缓存中间结果）统一写入：`/mnt/nfs/zijie/cd_circuit_output`。
5. 不改服务器全局配置，不影响其他项目。

### 0.2 重构完成标准（Definition of Done）

你可以把重构分成 4 层验收：

- L1（工程层）：新项目结构完整，`python -m` 与 CLI 脚本可运行。
- L2（算法层）：CD-T 核心传播与 source/target 分解实现完成。
- L3（实验层）：四类实验（greater-than, IOI, docstring, faithfulness）可通过脚本跑通。
- L4（质量层）：单测 + 集成测试 + 冒烟脚本通过，结果有固定输出目录和配置快照。

---

## 1. 一次性初始化（新项目目录）

> 说明：以下命令假定你自己执行。本文只给实施步骤，不会要求改全局 shell 配置。

### 1.1 创建项目目录

```bash
conda activate cdcircuit
mkdir -p ~/projects/Fuze_MI_repro/CD_Circuit-repro
cd ~/projects/Fuze_MI_repro/CD_Circuit-repro
```

### 1.2 复制依赖清单（沿用原始 requirements）

从旧项目复制：

```bash
cp /raid/home/zijie/projects/Fuze_MI_repro/CD_Circuit/requirements.txt ./requirements.txt
```

然后安装：

```bash
pip install -r requirements.txt
```

建议额外安装（仅在 `cdcircuit` 环境内）：

```bash
pip install pytest pytest-cov pyyaml typer rich
```

### 1.3 每次会话固定环境变量（仅当前 shell）

```bash
conda activate cdcircuit
cd ~/projects/Fuze_MI_repro/CD_Circuit-repro

export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/transformers

export CDC_OUTPUT_ROOT=/mnt/nfs/zijie/cd_circuit_output
export XDG_CACHE_HOME=/mnt/nfs/zijie/cd_circuit_output/.cache
export MPLCONFIGDIR=/mnt/nfs/zijie/cd_circuit_output/.mplconfig
export PIP_CACHE_DIR=/mnt/nfs/zijie/cd_circuit_output/.pip_cache

mkdir -p "$CDC_OUTPUT_ROOT" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
```

---

## 2. 推荐重构后的目录结构

目标结构（建议直接按此创建）：

```text
CD_Circuit-repro/
  README.md
  requirements.txt
  pyproject.toml
  Makefile
  .gitignore

  configs/
    base.yaml
    greater_than.yaml
    ioi.yaml
    docstring.yaml
    faithfulness.yaml

  src/
    cdcircuit/
      __init__.py
      version.py

      core/
        __init__.py
        types.py
        math_ops.py
        masks.py
        decomposition.py
        attention.py
        layer_ops.py

      models/
        __init__.py
        loader.py
        wrappers.py

      tasks/
        __init__.py
        greater_than/
          __init__.py
          dataset.py
          scoring.py
        ioi/
          __init__.py
          dataset.py
          scoring.py
        docstring/
          __init__.py
          prompts.py
          scoring.py

      algorithms/
        __init__.py
        source_to_target.py
        ablation.py
        faithfulness.py

      pipelines/
        __init__.py
        greater_than_pipeline.py
        ioi_pipeline.py
        docstring_pipeline.py
        faithfulness_pipeline.py

      io/
        __init__.py
        paths.py
        save.py
        logging.py
        config.py

      cli/
        __init__.py
        main.py
        run_greater_than.py
        run_ioi.py
        run_docstring.py
        run_faithfulness.py

  scripts/
    run_greater_than.sh
    run_ioi.sh
    run_docstring.sh
    run_faithfulness.sh
    smoke_all.sh

  tests/
    unit/
      test_math_ops.py
      test_masks.py
      test_decomposition_invariants.py
      test_attention_shapes.py
      test_ablation.py
      test_paths.py
    integration/
      test_greater_than_smoke.py
      test_ioi_smoke.py
      test_docstring_smoke.py
      test_faithfulness_smoke.py

  outputs/
    .gitkeep

  docs/
    How_TO_REBUILD.md
    ARCHITECTURE.md
    EXPERIMENTS.md
```

说明：
- 实际输出不写到仓库 `outputs/`，统一写到 `/mnt/nfs/zijie/cd_circuit_output`。
- `outputs/` 仅作占位防误写。

---

## 3. 模块职责设计（你要写哪些代码）

### 3.1 `core/`：算法算子层（最重要）

#### `src/cdcircuit/core/types.py`

实现内容：
- `Node`（layer_idx, sequence_idx, attn_head_idx）
- `AblationSet`（`tuple[Node, ...]`）
- `OutputDecomposition`
- `TargetNodeDecomposition`

你看完能得到：
- 全项目统一的数据结构契约。

#### `src/cdcircuit/core/math_ops.py`

实现内容：
- `normalize_rel_irrel(rel, irrel)`
- `prop_linear_core(rel, irrel, W, b)`
- `prop_activation(rel, irrel, act_fn)`
- `prop_layer_norm(rel, irrel, ln_module)`

关键要求：
- 每个函数都要可选开启断言：`assert_close(rel + irrel, total_ref)`。
- 数值稳定优先于微小性能收益。

#### `src/cdcircuit/core/masks.py`

实现内容：
- `get_extended_attention_mask(...)`
- decoder/encoder 分支处理

关键要求：
- 同时兼容 TransformerLens GPT 与 HF BERT。
- 写单测覆盖 mask shape 与 dtype。

#### `src/cdcircuit/core/attention.py`

实现内容：
- Q/K/V 分解传播
- attention scores / probs 分解
- head reshape 合并工具

关键要求：
- 明确区分“数学 patching”与“hook patching”。

#### `src/cdcircuit/core/layer_ops.py`

实现内容：
- `prop_gpt_layer(...)`
- `prop_bert_layer(...)`
- 预留 toy-model 分支

关键要求：
- 不要把 pipeline 逻辑混进来，保持纯函数式。

### 3.2 `models/`：模型加载与适配

#### `src/cdcircuit/models/loader.py`

实现内容：
- `load_gpt2_small(...)`
- `load_attn_only_4l(...)`
- `load_bert_like(...)`（预留）

必须做：
- 显式检查并记录 `HF_HOME`、`TRANSFORMERS_CACHE`。
- 记录模型加载参数到 run metadata。

#### `src/cdcircuit/models/wrappers.py`

实现内容：
- GPT/BERT wrapper（统一读取 Q/K/V/O、LayerNorm 参数）

### 3.3 `tasks/`：任务数据与评分

#### greater-than

文件：
- `tasks/greater_than/dataset.py`
- `tasks/greater_than/scoring.py`

实现内容：
- `YearDataset`（迁移原始逻辑）
- `get_valid_years`（迁移原始逻辑）
- 指标：正确年份 logit 差、head relevance 排序分数

#### ioi

文件：
- `tasks/ioi/dataset.py`
- `tasks/ioi/scoring.py`

实现内容：
- 迁移/包装 IOI dataset 构造
- 指标：`logits_to_ave_logit_diff`

#### docstring

文件：
- `tasks/docstring/prompts.py`
- `tasks/docstring/scoring.py`

实现内容：
- Prompt 生成器迁移
- 指标：目标 token 与干扰 token 的差异

### 3.4 `algorithms/`：核心算法与验证

#### `source_to_target.py`

实现内容：
- `prop_gpt(...)`
- `prop_bert(...)`
- `batch_run(...)`
- source->target 与 source->logits 统一接口

#### `ablation.py`

实现内容：
- `set_rel_at_source_nodes(...)`
- `patch_values(...)`
- `calculate_contributions(...)`

#### `faithfulness.py`

实现内容：
- mean ablation hook
- circuit keep-mask 生成
- faithfulness 计算函数

### 3.5 `pipelines/`：实验编排层

每个 pipeline 负责：
- 读 config
- 构建 run 目录
- 加载模型/数据
- 调用 algorithms
- 保存 artifacts（json/pt/png）

文件：
- `greater_than_pipeline.py`
- `ioi_pipeline.py`
- `docstring_pipeline.py`
- `faithfulness_pipeline.py`

### 3.6 `io/`：工程基础设施

#### `paths.py`

实现内容：
- 统一 run_id / run_dir 生成
- 强制输出根目录使用 `/mnt/nfs/zijie/cd_circuit_output`

#### `save.py`

实现内容：
- `save_json`
- `save_tensor`
- `save_figure`
- 自动创建目录、失败重试、路径打印

#### `config.py`

实现内容：
- YAML 读取
- CLI 覆盖参数
- 配置快照保存

#### `logging.py`

实现内容：
- 控制台 + 文件双日志
- 每个 run 单独日志文件

### 3.7 `cli/`：命令行入口

#### `cli/main.py`

推荐用 `typer` 或 `argparse` 实现子命令：
- `cdc run greater-than --config configs/greater_than.yaml`
- `cdc run ioi --config configs/ioi.yaml`
- `cdc run docstring --config configs/docstring.yaml`
- `cdc run faithfulness --config configs/faithfulness.yaml`

#### 单任务入口文件

用于简化调用与调试：
- `run_greater_than.py`
- `run_ioi.py`
- `run_docstring.py`
- `run_faithfulness.py`

---

## 4. 配置文件设计（建议模板）

### 4.1 `configs/base.yaml`

建议字段：

```yaml
seed: 42
device: cuda:0
output_root: /mnt/nfs/zijie/cd_circuit_output
hf_home: /mnt/nfs/zijie/huggingface_cache
log_level: INFO
num_workers: 0
save:
  save_intermediate: true
  save_plots: true
  save_tensors: true
```

### 4.2 任务配置示例 `configs/greater_than.yaml`

```yaml
inherits: base.yaml
task: greater_than
model:
  name: gpt2-small
  center_unembed: true
  center_writing_weights: true
  fold_ln: false
  refactor_factored_attn_matrices: true
dataset:
  start_year: 1000
  end_year: 1900
  n_samples: 5000
  balanced: true
  eos: true
algorithm:
  num_at_time: 64
  target_decomp_method: residual
  set_irrel_to_mean: true
experiment:
  top_k: 50
```

---

## 5. 脚本层设计（跑哪个部分用哪个命令）

### 5.1 Shell 脚本建议

#### `scripts/run_greater_than.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

conda activate cdcircuit
cd ~/projects/Fuze_MI_repro/CD_Circuit-repro

export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/transformers
export CDC_OUTPUT_ROOT=/mnt/nfs/zijie/cd_circuit_output

python -m cdcircuit.cli.main run greater-than --config configs/greater_than.yaml
```

同理写：
- `scripts/run_ioi.sh`
- `scripts/run_docstring.sh`
- `scripts/run_faithfulness.sh`

#### `scripts/smoke_all.sh`

顺序执行四个任务的 smoke 配置（小样本），用于 CI/本地快速验活。

### 5.2 Makefile 建议

```makefile
.PHONY: test lint smoke gt ioi doc faith

test:
	pytest -q

smoke:
	bash scripts/smoke_all.sh

gt:
	bash scripts/run_greater_than.sh

ioi:
	bash scripts/run_ioi.sh

doc:
	bash scripts/run_docstring.sh

faith:
	bash scripts/run_faithfulness.sh
```

---

## 6. 测试体系（重写时必须同步写）

你计划“重写所有 functions 和 test”，建议按三层测试：

### 6.1 单元测试（unit）

- `test_math_ops.py`
  - 验证 `rel + irrel` 不变量。
  - 验证 `normalize_rel_irrel` 在符号冲突时稳定。

- `test_masks.py`
  - encoder/decoder mask shape 测试。

- `test_attention_shapes.py`
  - Q/K/V reshape 和 attention 输出维度一致性。

- `test_ablation.py`
  - `set_rel_at_source_nodes` 是否正确改写指定节点。

### 6.2 集成测试（integration）

每个任务准备 tiny config（例如 8~16 样本）做 smoke：
- `test_greater_than_smoke.py`
- `test_ioi_smoke.py`
- `test_docstring_smoke.py`
- `test_faithfulness_smoke.py`

验证内容：
- 任务可跑完
- 输出目录和关键 artifacts 存在
- 指标值是有限数（非 NaN/Inf）

### 6.3 回归测试（可选但推荐）

固定随机种子与小批数据，记录关键中间张量统计（mean/std/top-k 排名）作为 baseline。
重构后每次改动比对偏差是否在阈值内。

---

## 7. 论文功能映射到重构代码（确保“覆盖原项目和论文”）

### 7.1 greater-than 论文线

原实现来源：
- `greater_than_task/*`
- `notebooks/greater_than_task.ipynb`

重构映射：
- dataset -> `tasks/greater_than/dataset.py`
- 评分 -> `tasks/greater_than/scoring.py`
- 主流程 -> `pipelines/greater_than_pipeline.py`
- CLI -> `run greater-than`

### 7.2 IOI 自动电路发现

原实现来源：
- `pyfunctions/ioi_dataset.py`
- `notebooks/IOI_automated_circuit_discovery.ipynb`

重构映射：
- dataset -> `tasks/ioi/dataset.py`
- source->target 分解 -> `algorithms/source_to_target.py`
- 排名和迭代 -> `pipelines/ioi_pipeline.py`

### 7.3 docstring 任务

原实现来源：
- `im_utils/*`
- `pyfunctions/toy_model.py`
- `notebooks/docstring_task_analysis.ipynb`

重构映射：
- prompt -> `tasks/docstring/prompts.py`
- toy propagation -> `core/layer_ops.py` + `algorithms/source_to_target.py`
- pipeline -> `pipelines/docstring_pipeline.py`

### 7.4 faithfulness 评估

原实现来源：
- `pyfunctions/faithfulness_ablations.py`
- `notebooks/faithfulness_eval.ipynb`

重构映射：
- hook/ablation -> `algorithms/faithfulness.py`
- 绘图与曲线 -> `pipelines/faithfulness_pipeline.py`

---

## 8. 推荐重构顺序（按周推进）

### 阶段 A：骨架与基础设施（第 1-2 天）

完成：
- 目录搭建
- `io/paths.py`, `io/config.py`, `io/save.py`, `io/logging.py`
- CLI 空壳命令
- `pytest` 可运行

验收：
- `python -m cdcircuit.cli.main --help`
- `pytest -q`（至少 3 个基础测试）

### 阶段 B：核心算子（第 3-4 天）

完成：
- `core/math_ops.py`, `core/masks.py`, `core/attention.py`, `core/types.py`

验收：
- 不变量测试通过
- shape 测试通过

### 阶段 C：主干算法（第 5-7 天）

完成：
- `algorithms/source_to_target.py`
- `algorithms/ablation.py`

验收：
- tiny 输入能跑出 decomposition 结果
- 可保存 `results.json` / `decomps.pt`

### 阶段 D：任务线逐个接入（第 2 周）

顺序建议：
1. greater-than
2. IOI
3. faithfulness
4. docstring

每接入一条任务线就补对应 smoke test。

---

## 9. 输出规范（强制）

每次运行产生：

```text
/mnt/nfs/zijie/cd_circuit_output/
  <run_id>/
    config_snapshot.yaml
    logs/
      run.log
    results/
      metrics.json
      top_heads.json
      decompositions.pt
    figures/
      *.png
      *.pdf
    artifacts/
      mean_acts.pt
      caches.pt
```

`run_id` 建议格式：
- `YYYYMMDD_HHMMSS_<task>_<model>`

---

## 10. 可执行命令清单（你最终应该能直接跑）

### 10.1 运行单任务

```bash
python -m cdcircuit.cli.main run greater-than --config configs/greater_than.yaml
python -m cdcircuit.cli.main run ioi --config configs/ioi.yaml
python -m cdcircuit.cli.main run docstring --config configs/docstring.yaml
python -m cdcircuit.cli.main run faithfulness --config configs/faithfulness.yaml
```

### 10.2 运行测试

```bash
pytest -q
pytest tests/unit -q
pytest tests/integration -q
```

### 10.3 运行冒烟全流程

```bash
bash scripts/smoke_all.sh
```

---

## 11. 每个关键文件的“最小可用实现”清单

下面是你在重构初期必须先写出来的 MVP 文件列表（按优先级）：

### P0（没有它就跑不起来）

- `src/cdcircuit/core/types.py`
- `src/cdcircuit/core/math_ops.py`
- `src/cdcircuit/core/masks.py`
- `src/cdcircuit/models/loader.py`
- `src/cdcircuit/algorithms/source_to_target.py`
- `src/cdcircuit/io/paths.py`
- `src/cdcircuit/io/config.py`
- `src/cdcircuit/io/save.py`
- `src/cdcircuit/cli/main.py`
- `src/cdcircuit/pipelines/greater_than_pipeline.py`

### P1（论文主功能需要）

- `src/cdcircuit/algorithms/ablation.py`
- `src/cdcircuit/algorithms/faithfulness.py`
- `src/cdcircuit/tasks/ioi/dataset.py`
- `src/cdcircuit/pipelines/ioi_pipeline.py`
- `src/cdcircuit/pipelines/faithfulness_pipeline.py`

### P2（完整覆盖原仓库）

- `src/cdcircuit/tasks/docstring/prompts.py`
- `src/cdcircuit/pipelines/docstring_pipeline.py`
- `tests/integration/test_docstring_smoke.py`

---

## 12. 与原仓库的一致性校验建议

为了确保“重构后仍对齐论文结果”，你可以做如下校验：

1. 在同样的 seed、小样本设置下，对比 top-k heads 排名重合度。
2. 对比 IOI 的 clean logit diff、null logit diff 方向是否一致。
3. 对比 faithfulness 曲线总体单调趋势与相对排序（CD-T > random）。
4. 对比关键中间张量统计量（mean/std）数量级。

不要求数值 bitwise 相等，但趋势和相对关系应稳定。

---

## 13. 典型风险与规避策略

### 13.1 风险：路径与缓存写错

规避：
- 程序启动时打印并 assert：
  - `HF_HOME` 前缀必须是 `/mnt/nfs/zijie/huggingface_cache`
  - 输出目录前缀必须是 `/mnt/nfs/zijie/cd_circuit_output`

### 13.2 风险：notebook 逻辑迁移遗漏

规避：
- 先把 notebook 按 cell 拆成 pipeline 步骤文档，再逐步落代码。
- 每迁移一段就补一个 unit 或 integration test。

### 13.3 风险：数值不稳定

规避：
- 在关键层后加可开关的 invariant 检查。
- 对 NaN/Inf 立即抛错并落盘上下文。

### 13.4 风险：工程过度复杂

规避：
- 先完成 P0/P1，再做优化。
- 不要在第一版就引入分布式、异步队列等额外复杂度。

---

## 14. 你可以直接执行的首日任务（Day-1 Checklist）

1. 建目录与 `pyproject.toml`。
2. 复制 `requirements.txt` 并安装。
3. 写 `io/paths.py` + `io/config.py` + `io/save.py`。
4. 写 `core/types.py` + `core/math_ops.py`（带 3 个基础测试）。
5. 写 `cli/main.py`（先只接 `run greater-than`）。
6. 写 `pipelines/greater_than_pipeline.py` 的空流程（可跑通到“加载配置并创建 run_dir”）。

完成标准：
- `python -m cdcircuit.cli.main run greater-than --config configs/greater_than.yaml` 能创建 run 目录和日志文件。

---

## 15. 你可以直接执行的首周交付（Week-1 Milestone）

交付目标：
- greater-than 全链路可跑
- unit 测试 >= 15 个
- integration smoke >= 1 个

必须产物：
- 可运行脚本 `scripts/run_greater_than.sh`
- 输出目录下有 `metrics.json` / `top_heads.json` / `decompositions.pt`
- `pytest -q` 通过

---

## 16. 后续建议（Week-2 ~ Week-3）

Week-2：
- 接入 IOI + faithfulness
- 完成 `scripts/run_ioi.sh` 与 `scripts/run_faithfulness.sh`

Week-3：
- 接入 docstring
- 完成回归测试与结果对照报告（`docs/EXPERIMENTS.md`）

---

## 17. 最终运行姿势（目标态）

你最终应只需要这类命令：

```bash
conda activate cdcircuit
cd ~/projects/Fuze_MI_repro/CD_Circuit-repro

export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/transformers
export CDC_OUTPUT_ROOT=/mnt/nfs/zijie/cd_circuit_output

bash scripts/run_greater_than.sh
bash scripts/run_ioi.sh
bash scripts/run_docstring.sh
bash scripts/run_faithfulness.sh
```

如果四条脚本都可稳定运行并产出标准化结果目录，说明你的“去 notebook 化重构”已经进入可维护状态。

---

## 18. 附：重构时如何对照旧代码

建议你在旧仓库与新仓库并行打开，按以下映射迁移：

- 旧：`pyfunctions/cdt_core.py` -> 新：`src/cdcircuit/core/math_ops.py` + `core/masks.py`
- 旧：`pyfunctions/cdt_source_to_target.py` -> 新：`src/cdcircuit/algorithms/source_to_target.py`
- 旧：`pyfunctions/cdt_ablations.py` -> 新：`src/cdcircuit/algorithms/ablation.py`
- 旧：`pyfunctions/faithfulness_ablations.py` -> 新：`src/cdcircuit/algorithms/faithfulness.py`
- 旧：`greater_than_task/*` -> 新：`src/cdcircuit/tasks/greater_than/*`
- 旧：`im_utils/*` + `pyfunctions/toy_model.py` -> 新：`src/cdcircuit/tasks/docstring/*` + `core/layer_ops.py`

迁移原则：
- 先“功能等价”，再“代码优雅”。
- 先“可测”，再“性能优化”。

---

## 19. 逐文件编写清单（每个文件具体写什么）

本节按你在第 2 节定义的目录逐文件给出“必须写的内容”。你可以把它当作实施 checklist。

### 19.1 根目录文件

#### `README.md`

至少包含：
- 项目目标（去 notebook 化 CD-T 重构）
- 快速开始（环境变量 + 一条示例命令）
- 支持任务列表（greater-than / IOI / docstring / faithfulness）
- 输出目录规范（`/mnt/nfs/zijie/cd_circuit_output/<run_id>`）
- 常见问题（缓存路径、OOM、导入路径）

#### `pyproject.toml`

至少包含：
- 项目元信息（name/version）
- `src` 布局声明
- entry point（可选）如 `cdc = cdcircuit.cli.main:app`
- `pytest` 配置（testpaths, pythonpath）

#### `Makefile`

至少目标：
- `test`、`smoke`、`gt`、`ioi`、`doc`、`faith`
- `fmt`（可选）、`lint`（可选）

#### `.gitignore`

至少忽略：
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`
- `.ipynb_checkpoints/`
- 本地中间输出（但不忽略文档）
- 不要忽略 `/mnt/nfs/zijie/...`（这是外部路径，不在仓库内）

### 19.2 `configs/`

#### `configs/base.yaml`

必须字段：
- `seed`, `device`
- `output_root`, `hf_home`
- `save.*`
- `log_level`

#### `configs/greater_than.yaml`

必须字段：
- 模型加载参数（gpt2-small + TransformerLens选项）
- 数据参数（year区间、N、balanced、eos）
- 算法参数（num_at_time、top_k、set_irrel_to_mean）

#### `configs/ioi.yaml`

必须字段：
- 数据参数（N、prompt_type、nb_templates）
- ablation 参数（mean reference、circuit keep策略）
- ranking 参数（top_k、迭代轮数）

#### `configs/docstring.yaml`

必须字段：
- 模型（attn-only-4l）
- prompt 生成参数（args 数、desc 长度、batch_size）
- 算法参数（target 方法、num_at_time）

#### `configs/faithfulness.yaml`

必须字段：
- clean/null 定义
- 节点增长策略（随机 / CDT）
- 输出图参数（dpi、格式）

### 19.3 `src/cdcircuit/__init__.py` 与 `version.py`

#### `src/cdcircuit/__init__.py`

写：
- `__version__` 导出
- 顶层包说明（1-2 行）

#### `src/cdcircuit/version.py`

写：
- `__version__ = "0.1.0"`（初始）
- 后续可接 git tag 自动化（非必须）

### 19.4 `src/cdcircuit/core/`

#### `src/cdcircuit/core/__init__.py`

写：
- 常用类型和核心函数再导出（便于 import）

#### `src/cdcircuit/core/types.py`

写：
- `Node`（NamedTuple/dataclass）
- `AblationSet = tuple[Node, ...]`
- `OutputDecomposition`（ablation_set + rel + irrel）
- `TargetNodeDecompositionList`（append/add）
- 必要的 `__repr__` 便于日志展示

#### `src/cdcircuit/core/math_ops.py`

写：
- `normalize_rel_irrel(rel, irrel)`
- `prop_linear_core(rel, irrel, W, b, tol=...)`
- `prop_linear(rel, irrel, linear_module)`
- `prop_activation(rel, irrel, act_fn)`
- `prop_layer_norm(rel, irrel, ln_module, tol=...)`
- `assert_invariant(rel, irrel, total_ref=None)`（调试开关）

要求：
- 文档字符串说明输入 shape 约定
- 关键步骤加 `torch.no_grad()` 兼容逻辑（由调用层控制）

#### `src/cdcircuit/core/masks.py`

写：
- `get_extended_attention_mask(attention_mask, input_shape, model, device)`
- encoder 与 decoder 分支
- TransformerLens 特判分支

#### `src/cdcircuit/core/decomposition.py`

写：
- 分解上下文对象（可选 dataclass）
- 公共 helper：
  - `split_rel_irrel_from_embedding(...)`
  - `merge_rel_irrel(...)`
  - `validate_decomposition(...)`

#### `src/cdcircuit/core/attention.py`

写：
- head 维度 reshape：
  - `reshape_separate_attention_heads`
  - `reshape_concatenate_attention_heads`
- attention 矩阵运算：
  - `mul_att(...)`
  - `prop_attention_probs(...)`
- query/key/value 路径分解 helper

#### `src/cdcircuit/core/layer_ops.py`

写：
- `prop_gpt_layer(...)`
- `prop_bert_layer(...)`
- `prop_toy_attn_only_layer(...)`（docstring）
- 每层返回：
  - `rel_out`, `irrel_out`
  - `layer_target_decomps`（如需要）

### 19.5 `src/cdcircuit/models/`

#### `src/cdcircuit/models/__init__.py`

写：
- 常用加载函数导出

#### `src/cdcircuit/models/loader.py`

写：
- `load_gpt2_small(device, **kwargs)`
- `load_attn_only_4l(device, **kwargs)`
- `load_model_from_config(cfg)`
- `verify_hf_env()`（检查缓存路径）

#### `src/cdcircuit/models/wrappers.py`

写：
- `GPTLayerNormWrapper`
- `GPTAttentionWrapper`
- 若需要，`BERTAttentionWrapper`

### 19.6 `src/cdcircuit/tasks/`

#### `src/cdcircuit/tasks/__init__.py`

写：
- 各任务入口导出

#### `src/cdcircuit/tasks/greater_than/__init__.py`

写：
- dataset/scoring 导出

#### `src/cdcircuit/tasks/greater_than/dataset.py`

写：
- `generate_real_sentence`, `generate_bad_sentence`
- `YearDataset`（迁移原逻辑）
- `get_valid_years`（可放 `utils`，但你当前结构里建议留在 task 内）

#### `src/cdcircuit/tasks/greater_than/scoring.py`

写：
- `compute_greater_than_score(...)`
- `rank_heads_by_relevance(...)`
- `build_candidate_circuit(top_k, ranked_heads, seq_len)`

#### `src/cdcircuit/tasks/ioi/__init__.py`

写：
- dataset/scoring 导出

#### `src/cdcircuit/tasks/ioi/dataset.py`

写：
- `build_ioi_dataset(...)`
- `build_abc_reference_dataset(...)`
- 需要时封装旧 `IOIDataset` 的兼容层

#### `src/cdcircuit/tasks/ioi/scoring.py`

写：
- `logits_to_ave_logit_diff(...)`
- `compute_ioi_metrics(...)`（clean/null/ablated）

#### `src/cdcircuit/tasks/docstring/__init__.py`

写：
- prompts/scoring 导出

#### `src/cdcircuit/tasks/docstring/prompts.py`

写：
- `Prompt` dataclass
- `docstring_prompt_gen(...)`
- `build_prompt_batch(...)`

#### `src/cdcircuit/tasks/docstring/scoring.py`

写：
- `compute_docstring_logit_gap(...)`
- `filter_good_prompts(...)`

### 19.7 `src/cdcircuit/algorithms/`

#### `src/cdcircuit/algorithms/__init__.py`

写：
- 三大算法模块导出

#### `src/cdcircuit/algorithms/source_to_target.py`

写：
- `prop_gpt(...)`（核心）
- `prop_bert(...)`（核心）
- `prop_toy_model(...)`（docstring）
- `batch_run(prop_fn, ablation_sets, num_at_time=...)`
- `collect_target_decompositions(...)`
- `compute_logits_decomposition_scores(...)`

#### `src/cdcircuit/algorithms/ablation.py`

写：
- `set_rel_at_source_nodes(...)`
- `calculate_contributions(...)`
- `patch_values(...)`
- `make_ablation_sets(...)`（辅助）

#### `src/cdcircuit/algorithms/faithfulness.py`

写：
- `compute_means_by_template(...)`
- `add_mean_ablation_hook(...)`
- `compute_faithfulness(circuit_score, model_clean, model_null)`
- `evaluate_circuit_faithfulness(...)`

### 19.8 `src/cdcircuit/pipelines/`

#### `src/cdcircuit/pipelines/__init__.py`

写：
- 4 条 pipeline 入口导出

#### `src/cdcircuit/pipelines/greater_than_pipeline.py`

写：
- `run(cfg)` 主函数
- 流程：
  - set seed
  - create run_dir
  - load model
  - load/generate dataset
  - compute mean acts
  - source->logits decomposition
  - rank + save results

输出至少包含：
- `metrics.json`
- `top_heads.json`
- `decompositions.pt`

#### `src/cdcircuit/pipelines/ioi_pipeline.py`

写：
- `run(cfg)` 主函数
- 流程：
  - IOI + ABC dataset
  - decomposition ranking
  - 迭代 target 扩展（可选）
  - 保存各轮结果

#### `src/cdcircuit/pipelines/docstring_pipeline.py`

写：
- `run(cfg)` 主函数
- 流程：
  - prompt 生成
  - 过滤可用样本
  - toy model decomposition
  - 保存 ranking 与中间结果

#### `src/cdcircuit/pipelines/faithfulness_pipeline.py`

写：
- `run(cfg)` 主函数
- 流程：
  - clean/null 基线
  - 随机电路曲线
  - CDT 电路曲线
  - 生成对比图并保存

### 19.9 `src/cdcircuit/io/`

#### `src/cdcircuit/io/__init__.py`

写：
- io 工具再导出

#### `src/cdcircuit/io/paths.py`

写：
- `build_run_id(task, model)`
- `build_run_dir(output_root, run_id)`
- `ensure_run_layout(run_dir)`（创建 logs/results/figures/artifacts）
- `assert_safe_output_root(path)`（必须在 `/mnt/nfs/zijie/cd_circuit_output` 下）

#### `src/cdcircuit/io/save.py`

写：
- `save_json(path, obj)`
- `save_tensor(path, tensor_or_obj)`
- `save_figure(path, fig_or_plt)`
- `save_config_snapshot(path, cfg)`

#### `src/cdcircuit/io/logging.py`

写：
- `setup_logger(run_dir, level)`
- 控制台 + 文件 handler
- 每条 pipeline 统一 logger 命名

#### `src/cdcircuit/io/config.py`

写：
- `load_yaml(path)`
- `resolve_inheritance(cfg)`（支持 `inherits`）
- `apply_cli_overrides(cfg, overrides)`
- `validate_required_fields(cfg, task)`

### 19.10 `src/cdcircuit/cli/`

#### `src/cdcircuit/cli/__init__.py`

写：
- CLI app 导出（可选）

#### `src/cdcircuit/cli/main.py`

写：
- 主入口 app（`argparse` 或 `typer`）
- 子命令：
  - `run greater-than`
  - `run ioi`
  - `run docstring`
  - `run faithfulness`
- 每个命令最终调用对应 pipeline `run(cfg)`

#### `src/cdcircuit/cli/run_greater_than.py` 等单任务入口

写：
- 解析参数并调用 pipeline
- 保留给单步调试和以后集成系统调用

### 19.11 `scripts/`

#### `scripts/run_greater_than.sh`

写：
- 激活 `cdcircuit`
- 导出缓存/输出环境变量
- 调用 CLI 命令

#### `scripts/run_ioi.sh`

写：
- 同上，调用 `run ioi`

#### `scripts/run_docstring.sh`

写：
- 同上，调用 `run docstring`

#### `scripts/run_faithfulness.sh`

写：
- 同上，调用 `run faithfulness`

#### `scripts/smoke_all.sh`

写：
- 顺序跑四个 smoke 配置
- 任一失败立即退出

### 19.12 `tests/unit/`

#### `tests/unit/test_math_ops.py`

写：
- 线性层分解不变量测试
- LayerNorm 分解数值稳定测试
- activation 分解测试

#### `tests/unit/test_masks.py`

写：
- decoder mask shape 测试
- encoder mask shape 测试

#### `tests/unit/test_decomposition_invariants.py`

写：
- 多层串联后 `rel+irrel` 仍可重构总量
- `normalize_rel_irrel` 在随机张量上稳定

#### `tests/unit/test_attention_shapes.py`

写：
- reshape 分离/合并后形状一致
- attention 乘法输出 shape 正确

#### `tests/unit/test_ablation.py`

写：
- source node patch 只改目标节点
- 非目标节点值保持不变

#### `tests/unit/test_paths.py`

写：
- run_dir 构造规范
- 输出根目录安全检查

### 19.13 `tests/integration/`

#### `tests/integration/test_greater_than_smoke.py`

写：
- 用 tiny config 跑 `greater-than`
- assert 输出文件存在且可读取

#### `tests/integration/test_ioi_smoke.py`

写：
- 用 tiny config 跑 IOI
- assert `metrics.json` 有 `clean` / `ablated` 等关键字段

#### `tests/integration/test_docstring_smoke.py`

写：
- 跑 docstring pipeline
- assert ranking 结果非空

#### `tests/integration/test_faithfulness_smoke.py`

写：
- 跑 faithfulness pipeline
- assert 图像与曲线数据生成成功

### 19.14 `docs/`

#### `docs/ARCHITECTURE.md`

写：
- 分层架构图（core/algorithms/tasks/pipelines/io/cli）
- 数据流图（输入 -> 分解 -> 评分 -> 输出）

#### `docs/EXPERIMENTS.md`

写：
- 每个任务一组标准命令
- 关键参数解释
- baseline 结果记录模板

#### `docs/How_TO_REBUILD.md`

写：
- 与本文一致（可以软链接或同步维护）

### 19.15 每个文件的统一代码规范（建议）

建议你给所有 `src/cdcircuit/**/*.py` 遵循：
- 顶部模块 docstring（说明该文件职责）
- 关键公开函数有完整 docstring：
  - 输入 shape/类型
  - 返回值
  - 抛错条件
- 不要在底层函数里写硬编码绝对路径
- 路径全部通过 `io/paths.py` 和 config 注入
- 所有随机流程先 `set_seed`
- 所有实验入口都落盘 `config_snapshot.yaml`
