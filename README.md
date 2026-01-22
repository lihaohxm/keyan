# keyan

本仓库提供多 RIS 下行语义通信的全 Python 仿真框架，默认可在 PyCharm 或命令行一键运行。

## 目录结构

```
results/   # 运行结果输出（.mat/.json/.csv）
figures/   # 画图输出
logs/      # 运行日志
scripts/   # 保存/汇总/画图脚本
matlab/    # MATLAB 绘图脚本（可选）
sim/       # 核心仿真模块
experiments/ # 仿真实验入口
```

## 依赖

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 运行方式（建议命令）

单次仿真（固定 P）：

```bash
python -m experiments.run_once --seed 1 --mc 20 --p_dbw -5 --semantic_mode proxy
```

扫功率生成曲线：

```bash
python -m experiments.sweep --mc 100 --semantic_mode proxy --x p_dbw
```

扫权重生成 Pareto：

```bash
python -m experiments.pareto --mc 100 --semantic_mode proxy
```

画图（自动选择最新 run）：

```bash
python scripts/plot_curves_py.py --latest
```

### DeepSC 查表模式

`semantic_mode=table` 时需要提供表格路径。支持 CSV/NPZ/MAT。

**CSV 长表格式（推荐）**：

```csv
snr_db,M,xi
-10,8,0.12
-10,16,0.18
-5,8,0.25
```

**CSV 网格格式**（首列为 snr_db 或 gamma，其他列为 M 值）：

```csv
snr_db,8,16,32
-10,0.12,0.18,0.23
-5,0.25,0.3,0.35
```

运行示例：

```bash
python -m experiments.run_once --semantic_mode table --semantic_table path/to/table.csv
```

`semantic_mode=deepsc_stub` 为预留接口，当前等价于 proxy。

## 仿真输出

每次运行将写入：

- `results/<run_id>_curves.mat`
- `results/<run_id>_metrics.json`
- `results/<run_id>_curves.csv`

Python 画图脚本会输出三类图到 `figures/`：

1. sum-rate vs x
2. avg QoE vs x
3. Pareto：sum-rate vs avg QoE

## 保存结果接口（Python）

`scripts/save_results.py` 提供 `save_run` 用于保存单次实验：

```python
from scripts.save_results import save_run

run_id = "demo_run"
x_axis = [-10, -5, 0]
metrics_dict = {"runtime_s": 12.3, "seed": 42}
arrays_dict = {
    "sum_rate_random": [10, 12, 15],
    "avg_qoe_random": [0.8, 0.6, 0.5],
}

save_run(run_id, x_axis, metrics_dict, arrays_dict)
```

## MATLAB 绘图（可选）

`matlab/plot_curves.m` 会读取 `results/<run_id>_curves.mat`，并保存到 `figures/`。

```matlab
% 画指定 run_id
plot_curves("demo_run");

% 自动扫描 results/ 下所有 *_curves.mat
plot_curves();
```
