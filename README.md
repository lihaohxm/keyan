# keyan

## 实验结果目录规范

```
results/  # 运行结果输出（.mat/.json/.csv）
figures/  # 画图输出（例如 BER/BLER 曲线）
logs/     # 运行日志
scripts/  # 保存结果的脚本
matlab/   # MATLAB 绘图脚本
```

## 保存结果（Python）

`scripts/save_results.py` 提供 `save_run` 用于保存单次实验：

```python
from scripts.save_results import save_run

run_id = "demo_run"
snr_db = [-2, 0, 2, 4]
metrics_dict = {"runtime_s": 12.3, "seed": 42}
arrays_dict = {
    "ber": [1e-1, 5e-2, 2e-2, 8e-3],
    "bler": [0.6, 0.35, 0.15, 0.05],
}

save_run(run_id, snr_db, metrics_dict, arrays_dict)
```

输出文件：

- `results/<run_id>_curves.mat`（`scipy.io.savemat`）
- `results/<run_id>_metrics.json`
- `results/<run_id>_curves.csv`

## 绘图（MATLAB）

`matlab/plot_curves.m` 会读取 `results/<run_id>_curves.mat` 中的 `snr_db`、`ber`、`bler`
并保存图片到 `figures/`：

```matlab
% 画指定 run_id
plot_curves("demo_run");

% 自动扫描 results/ 下所有 *_curves.mat
plot_curves();
```

## 多 RIS 语义通信仿真框架

新增的 `sim/` 与 `experiments/` 提供多 RIS 辅助下行语义通信的匹配与 QoE 对比仿真。输出结果复用 `save_run` 落盘到 `results/`，并支持 Python 画图。

### 快速开始（可复制命令）

1. 扫功率生成 curves + metrics：

```bash
python -m experiments.sweep --x P_dBW --x_list "-10,-5,0" --mc 100 --seed 1 --algos "random,norm,qoe,exhaustive"
```

2. 生成 Pareto：

```bash
python -m experiments.pareto --mc 100 --seed 1 --weights "0.8,0.2;0.5,0.5;0.2,0.8" --algos "random,norm,qoe"
```

3. 画图（Python）：

```bash
python scripts/plot_curves_py.py --run_id <run_id>
```

### 结果输出

运行后会在 `results/` 下生成：

- `<run_id>_metrics.json`：包含配置、算法、均值/方差/置信区间、Pareto 点等。
- `<run_id>_curves.csv`：横轴 + 每算法的 sum-rate 与 avg QoE 曲线。
- `<run_id>_curves.mat`：与 CSV 字段一致，便于 MATLAB 读取。

Python 画图脚本会输出三类图到 `figures/`：

1. sum-rate vs x
2. avg QoE vs x
3. Pareto：sum-rate vs avg QoE
