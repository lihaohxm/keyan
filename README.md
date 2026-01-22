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
