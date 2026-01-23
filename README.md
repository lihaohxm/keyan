# keyan

本仓库提供多 RIS 下行语义通信仿真框架：**MATLAB 负责主仿真**（几何/信道/匹配/SINR/QoE/扫参/画图/落盘），**Python 仅负责离线生成 DeepSC 语义映射表**供 MATLAB 查表插值。

## 目录结构

```
matlab_sim/          % MATLAB 仿真核心
matlab_sim/matching/ % RIS 匹配策略
matlab_experiments/  % MATLAB 实验入口（run_once/sweep/pareto）
matlab_scripts/      % MATLAB 结果保存与画图
python_deepsc/       % Python 导出语义映射表（stub）
semantic_tables/     % DeepSC 导出的 CSV 表
results/             % 运行结果输出（.mat/.json/.csv）
figures/             % 画图输出
logs/                % 运行日志（预留）
```

## Python 依赖（仅用于导出语义表）

```bash
pip install -r requirements.txt
```

## MATLAB 运行方式（推荐）

在 MATLAB 命令行（仓库根目录）执行：

**单次仿真**：

```matlab
matlab_experiments.run_once('seed',1,'mc',20,'p_dbw',-5,'semantic_mode','proxy')
```

**查表模式**：

```matlab
matlab_experiments.run_once('seed',1,'mc',20,'p_dbw',-5,'semantic_mode','table','table_path','semantic_tables/deepsc_table.csv')
```

**扫参**：

```matlab
matlab_experiments.sweep('seed',1,'mc',100,'semantic_mode','proxy','x','p_dbw')
```

**Pareto 扫权重**：

```matlab
matlab_experiments.pareto('seed',1,'mc',100,'semantic_mode','proxy')
```

**画图**：

```matlab
matlab_scripts.plot_curves('latest',true)
```

## Python 生成语义映射表

`python_deepsc/export_semantic_table.py` 默认生成长表格式 CSV：

```bash
python python_deepsc/export_semantic_table.py --out semantic_tables/deepsc_table.csv
```

表格格式（长表）：

```csv
snr_db,M,xi
-10,8,0.12
-10,16,0.18
-5,8,0.25
```

MATLAB 端 `semantic_map` 会读取 CSV 并做二维插值，输入 `snr_db=10*log10(gamma)` 与 `M`，输出 `xi`。

## 结果输出

每次运行将写入：

- `results/<run_id>_curves.mat`
- `results/<run_id>_metrics.json`
- `results/<run_id>_curves.csv`

MATLAB 画图脚本会输出三类图到 `figures/`：

1. sum-rate vs x
2. avg QoE vs x
3. Pareto：sum-rate vs avg QoE
