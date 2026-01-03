# 平台说明（统一配置与作业）

本仓库包含多个模块：SISSO 描述符、机器学习建模、高通量计算与 VASP 工作流，已集中使用共享的配置与 SLURM 模板。

## 关键目录
- `services/config/`: 全局配置 `default_config.json`；`loader.py` 为统一读取入口（集群/队列/模板/远程账号）；`manager.py` 提供默认配置与旧接口兼容（`config/` 仍保留薄包装）。
- `services/`: 通用服务（SSH、SISSO、VASP 等）与各模块模板（SISSO、VASP、ML 自带各自的 slurm 脚本）。
- `machine_learning/`: ML Dash 应用、作业生成与训练逻辑。
- `discriptor/`: SISSO 描述符 Dash 应用。
- `high-throught-calcultion/`: 高通量相关脚本与界面。

## 配置与模板
- `services/config/default_config.json` 使用 `active_cluster` 选择集群；每个 `clusters.<name>` 可指定 `remote_server`、`queue`（partition/time/nodes/ntasks_per_node）、`templates.slurm` 等。
- SLURM 模板随模块提供：SISSO 在 `services/sisso/templates/slurm.sh`，机器学习在 `services/machine_learning/slurm.sh`，VASP 在 `services/vasp/templates/slurm.sh`。如需全局自定义，请在配置中显式填写模板路径。

## 统一输出目录
- `services/config/default_config.json` 的 `local_paths.results_root` 默认为 `./outputs`，所有模块共享此根目录，可通过环境变量 `RESULTS_ROOT` 或覆盖配置修改。
- 机器学习：`outputs/machine_learning`（训练日志、指标、模型文件）。
- 描述符（SISSO）：`outputs/discriptor`（下载的 `results.csv`）。
- 高通量/DFT：`outputs/high_throughput`（下载的 `CONTCAR` 等结构文件）。

## 运行示例
- `python discriptor/app.py`
- `python machine_learning/app.py`
- `python high-throught-calcultion/app.py`

### 新增：Django 后端 + Docker
- Django 后端放在 `backend/`，提供统一的 `/api/jobs/`（提交、查询）接口，复用 `services` 中的 SSH/SLURM 逻辑。
- Docker 一键启动：`docker-compose up --build`（后端 8000，Dash 默认 8050）。
- Dash 端如果检测到环境变量 `BACKEND_BASE_URL`（Compose 已注入），会自动改为通过 Django 提交作业；未设置时回退原有直连 SSH。

## 作业提交要点
- ML/SISSO 使用共享 SSH 与 SLURM 模板。若切换集群或队列，请先更新 `services/config/default_config.json` 中的对应集群配置或 `active_cluster`。
- 如需固定使用模块内模板，可在配置里将 `templates.slurm` 指向模块目录下的模板文件。

## 维护
- 旧的分散说明与测试脚本已清理。如需新增文档，请集中维护在本文件或单独子目录中。
