# SparseDC configs 配置学习笔记

本文整理 `configs/` 文件夹下的主要内容，帮助快速理解 Hydra 配置如何组织训练、评测、数据、模型和日志输出。

## 1. 配置目录总览

```text
configs/
├── train.yaml        # 训练入口配置
├── eval.yaml         # 评测入口配置
├── callbacks/        # Lightning callbacks
├── data/             # 数据集与 DataModule 配置
├── debug/            # 调试模式配置
├── experiment/       # 实际实验配置，覆盖默认 data/model/hparams
├── extras/           # 运行前辅助行为，如打印配置、warnings、tags
├── hparams/          # 全局超参默认值
├── hydra/            # Hydra 输出目录与日志配置
├── logger/           # Lightning logger 配置
├── model/            # 模型、优化器、scheduler、metric 配置
├── paths/            # 数据、日志、输出路径配置
└── trainer/          # Lightning Trainer 配置
```

这个项目依赖 Hydra 的 defaults 列表组合配置。最重要的原则是：以 `experiment/*.yaml` 中的覆盖结果为准，尤其是模型结构、数据集选择、`max_depth`、`batch_size` 等。

## 2. 顶层入口配置

### 2.1 `train.yaml`

`train.yaml` 是 `python train.py ...` 使用的根配置。

默认组合顺序：

```yaml
defaults:
  - _self_
  - hparams: default
  - data: nyu
  - model: default
  - callbacks: default
  - logger: tensorboard
  - trainer: ddp
  - paths: default
  - extras: default
  - hydra: default
  - experiment: null
  - optional local: default
  - debug: null
```

主要字段：

- `mode: train`：用于 Hydra 输出目录。
- `task_name: train`：默认任务名，通常会被 experiment 覆盖。
- `tags: ["dev"]`：实验标签。
- `train: True`：是否执行 `trainer.fit`。
- `test: True`：训练后是否使用 best checkpoint 执行 test。
- `ckpt_path: null`：恢复训练 checkpoint 路径。

注意：当前仓库没有可用的 `configs/model/default.yaml` 和 `configs/experiment/default.yaml`。不要裸跑 `python train.py`，应使用：

```bash
python train.py experiment=final_version
python train.py experiment=final_version_kitti
```

### 2.2 `eval.yaml`

`eval.yaml` 是 `python eval.py ...` 使用的根配置。

默认组合：

```yaml
defaults:
  - _self_
  - data: nyu
  - model: default
  - logger: tensorboard
  - trainer: gpu
  - paths: default
  - extras: default
  - hydra: default
  - hparams: default
  - experiment: default
```

主要字段：

- `mode: eval`：用于输出目录。
- `task_name: eval`：默认任务名。
- `seed: 2023`：评测随机种子。
- `ckpt_path: ???`：必须从命令行或脚本传入。

实际评测应通过脚本或显式覆盖 experiment，例如：

```bash
./eval_nyu.sh final_version final_version pretrain/nyu.ckpt
./eval_kitti.sh final_version_kitti final_version_kitti_test pretrain/kitti.ckpt
./eval_sunrgbd.sh final_version final_version pretrain/nyu.ckpt
```

## 3. 全局超参 `hparams/default.yaml`

该文件使用 `# @package _global_`，字段直接放到全局配置根节点。

主要字段：

- `batch_size: 8`：默认 batch size。
- `num_workers: 8`：默认 DataLoader worker 数。
- `seed: 2023`：随机种子。
- `monitor: RMSE`：checkpoint、early stopping、scheduler 监控指标。
- `num_sample: 500`：NYU/SUNRGBD 稀疏采样点数。
- `base_lr: 0.001`：默认学习率，常被 experiment 改成 `0.0001`。
- `max_depth: 10.0`：默认最大深度，KITTI 会覆盖为 `80.0`。
- `is_sparse: True`：训练时是否启用随机稀疏增强。
- `is_fill: False`：全局默认 fill 开关，但实际模型通常在 experiment 的 `model.net.is_fill` 中设置。

常用命令行覆盖：

```bash
python train.py experiment=final_version num_sample=100
python train.py experiment=final_version is_sparse=false
python train.py experiment=final_version base_lr=0.00005
```

## 4. 路径配置 `paths/default.yaml`

主要字段：

- `root_dir: .`：项目根目录。
- `data_dir: /home/an/Desktop/SparseDC/data`：数据根目录，当前是绝对路径。
- `log_dir: ${paths.root_dir}/logs`：日志根目录。
- `output_dir: ${hydra:runtime.output_dir}`：Hydra 每次运行动态输出目录。
- `work_dir: ${hydra:runtime.cwd}`：启动命令时的工作目录。

换机器时优先从命令行覆盖数据目录：

```bash
python train.py experiment=final_version paths.data_dir=/your/data/root
```

数据集实际路径由 `data/*.yaml` 拼接：

- NYU：`${paths.data_dir}/nyudepthv2`
- KITTI：`${paths.data_dir}/kitti_depth` 和 `${paths.data_dir}/kitti_raw`
- SUNRGBD：`${paths.data_dir}/SUNRGBD`

## 5. Hydra 输出配置 `hydra/default.yaml`

该文件控制 Hydra 日志和输出目录。

运行输出目录：

```text
${paths.log_dir}/${hydra.runtime.choices.data}/${hydra.runtime.choices.model}/${mode}/${task_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
```

展开后类似：

```text
logs/nyu/Uncertainty/train/resnet18_pvt_v2_b1_final_version/2026-05-03_12-00-00/
```

多次 sweep 输出目录：

```text
${paths.log_dir}/${hydra.runtime.choices.data}/${hydra.runtime.choices.model}/${mode}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}/${hydra.job.num}
```

## 6. 数据配置 `data/`

所有数据配置都实例化同一个类：

```yaml
_target_: src.data.DataModule
```

`DataModule` 内部根据 `dataset` 选择具体 Dataset。

### 6.1 `data/nyu.yaml`

主要字段：

- `dataset: nyu`
- `batch_size: ${batch_size}`
- `num_workers: ${num_workers}`
- `args.augment: true`
- `args.data_dir: ${paths.data_dir}/nyudepthv2`
- `args.split_json: nyu.json`
- `args.num_sample: ${num_sample}`
- `args.is_sparse: ${is_sparse}`

作用：读取 NYU Depth V2，训练阶段可随机改变稀疏采样点数。

### 6.2 `data/kitti.yaml`

主要字段：

- `dataset: kitti`
- `args.data_folder: ${paths.data_dir}/kitti_depth`
- `args.data_folder_rgb: ${paths.data_dir}/kitti_raw`
- `args.val: full`
- `args.val_h: 352`
- `args.val_w: 1216`
- `args.use_rgb: true`
- `args.use_d: true`
- `args.random_crop_height: 320`
- `args.random_crop_width: 1216`
- `args.is_sparse: ${is_sparse}`

`args.val` 很关键：

- `full`：使用 KITTI train/val full 路径。
- `select`：使用 `data_depth_selection/val_selection_cropped`，通常配合 `num_lines` 做 lines64/32/16/8/4 评测。

### 6.3 `data/sunrgbd.yaml`

主要字段：

- `dataset: sunrgbd`
- `args.data_dir: ${paths.data_dir}/SUNRGBD`
- `args.num_sample: ${num_sample}`
- `args.radio: 0.2`

`radio` 用于从训练文件中划分验证集比例。

## 7. 模型配置 `model/Uncertainty.yaml`

该文件定义 LightningModule、网络、优化器、scheduler 和 metric。

顶层模型：

```yaml
_target_: src.models.model.DepthLitModule
```

主要字段：

- `is_warmup: false`：默认不 warmup，实验配置通常改成 `true`。
- `monitor: ${monitor}`：默认监控 `RMSE`。
- `base_lr: ${base_lr}`：传给优化器和 warmup。
- `save_dir: ${paths.output_dir}`：结果输出目录。
- `dataset: ${data.dataset}`：用于区分 NYU/KITTI/SUNRGBD 输出逻辑。

网络部分：

```yaml
net:
  _target_: src.models.base.Uncertainty_
  backbone_g:
    _target_: src.models.backbones.PVTV2
  backbone_l:
    _target_: src.models.backbones.ResNetU_
    model_name: resnet18
  decode:
    _target_: src.models.decodes.UncertaintyFuse_
    max_depth: ${max_depth}
  criterion:
    _target_: src.criterion.loss.DepthLoss
  channels: 64
  is_padding: true
  padding_size: [256, 320]
  max_depth: ${max_depth}
```

默认 `Uncertainty.yaml` 是基础骨架，实际可运行配置会在 `experiment/*.yaml` 中补齐 `decode` 的通道参数、`refiner`、`is_fill`、PVT 预训练权重等。

优化器与调度器：

- `optimizer`: `torch.optim.Adam`，`lr=${base_lr}`，`weight_decay=0.00001`。
- `scheduler`: `torch.optim.lr_scheduler.ReduceLROnPlateau`，监控 `val/${monitor}`。

指标：

- `metric`: `src.criterion.metric.DepthCompletionMetric`，使用 `${max_depth}`。

## 8. 实验配置 `experiment/`

`experiment/` 是最重要的配置目录。它通过 `defaults.override` 覆盖根配置中的 data/model，并补齐模型结构。

### 8.1 `final_version.yaml`

NYU 主实验配置。

关键设置：

- 覆盖模型为 `Uncertainty`。
- `batch_size: 8`
- `num_workers: 24`
- `max_depth: 10.0`
- `base_lr: 0.0001`
- `model.is_warmup: true`
- `model.net.is_fill: true`
- global branch：`PVTV2`，`model_name: pvt_v2_b1`，加载 `pretrain/pvt_v2_b1.pth`。
- local branch：`ResNetU_`，`model_name: resnet18`。
- decode：`UncertaintyFuse_`，`is_gate_fuse: true`。
- criterion：`1.0*L2+1.0*L1`。
- refiner：`NLSPN`，`prop_time: 18`，`affinity: TGASS`。
- `task_name`: `resnet18_pvt_v2_b1_final_version` 形式。

### 8.2 `final_version_kitti.yaml`

KITTI 训练配置。

相对 NYU 的主要变化：

- 覆盖数据为 `kitti`。
- `batch_size: 2`
- `num_workers: 16`
- `data.args.num_lines: lines64`
- `max_depth: 80.0`
- `model.net.is_padding: false`
- `model.net.padding_size: [256, 1280]`
- scheduler patience 更短：`patience: 3`。

### 8.3 `final_version_kitti_test.yaml`

KITTI 评测配置。

主要变化：

- 覆盖数据为 `kitti`。
- `batch_size: 1`
- `num_workers: 2`
- `data.args.val: select`
- `data.args.num_lines: lines64`
- `max_depth: 80.0`

该配置通常配合 `eval_kitti.sh` 使用。若评测 lines32/16/8/4，通常覆盖 `data.args.num_lines` 或由脚本循环传入。

### 8.4 `final_version_sunrgbd.yaml`

SUNRGBD 配置。

主要变化：

- 覆盖数据为 `sunrgbd`。
- `batch_size: 2`
- `num_workers: 8`
- `max_depth: 10.0`
- 模型结构基本沿用 NYU final version。
- `task_name` 后缀为 `_sunrgbd`。

### 8.5 `final_version_kitti_no_sparse.yaml`

KITTI 的 no sparse ablation。

主要变化：

- 覆盖数据为 `kitti`。
- `data.args.val: select`
- `is_sparse: false`
- `task_name` 后缀为 `_no_sparse`。

### 8.6 `no_fill.yaml`

消融 SFFM/fill 的 NYU 配置。

关键变化：

- `model.net.is_fill: false`
- `backbone_g.is_fill: false`
- `backbone_l.is_fill: false`
- 其他主干、decode、refiner 基本保持不变。

### 8.7 `no_sparse.yaml`

关闭稀疏增强的 NYU 配置。

关键变化：

- `is_sparse: false`
- 模型仍启用 `is_fill: true`。

### 8.8 `no_fuse.yaml`

关闭不确定度 gate fuse 的 NYU 配置。

关键变化：

- `model.net.decode.is_gate_fuse: false`
- 其他结构基本与 `final_version.yaml` 相同。

## 9. Trainer 配置 `trainer/`

### 9.1 `trainer/default.yaml`

基础 Lightning Trainer 配置：

- `_target_: pytorch_lightning.Trainer`
- `default_root_dir: ${paths.output_dir}`
- `min_epochs: 1`
- `max_epochs: 500`
- `accelerator: cpu`
- `devices: 1`
- `check_val_every_n_epoch: 1`
- `deterministic: False`
- `log_every_n_steps: 5`

### 9.2 `trainer/gpu.yaml`

继承 default，改为单卡 GPU：

```yaml
accelerator: gpu
devices: 1
```

单卡调试常用：

```bash
python train.py experiment=final_version trainer=gpu trainer.devices=1
```

### 9.3 `trainer/ddp.yaml`

继承 default，改为多卡 DDP spawn：

```yaml
strategy: ddp_spawn
accelerator: gpu
devices: -1
num_nodes: 1
sync_batchnorm: True
auto_select_gpus: True
```

`train.yaml` 默认使用 `trainer: ddp`。如果机器没有多卡或只想快速调试，应显式覆盖为 `trainer=gpu`。

### 9.4 `trainer/cpu.yaml`

继承 default，强制 CPU：

```yaml
accelerator: cpu
devices: 1
```

## 10. Callback 配置 `callbacks/`

### 10.1 `callbacks/default.yaml`

组合以下 callback：

- `model_checkpoint.yaml`
- `early_stopping.yaml`
- `model_summary.yaml`
- `rich_progress_bar.yaml`
- `learning_rate_monitor.yaml`

并覆盖默认参数：

- checkpoint 保存到 `${paths.output_dir}/checkpoints`
- checkpoint 文件名为 `epoch_{epoch:03d}`
- checkpoint 监控 `val/${monitor}`，默认 `val/RMSE`
- `mode: min`
- `save_last: True`
- early stopping patience 为 `10`
- model summary `max_depth: -1`
- learning rate monitor 按 step 记录

### 10.2 子配置说明

- `model_checkpoint.yaml`：实例化 `pytorch_lightning.callbacks.ModelCheckpoint`。
- `early_stopping.yaml`：实例化 `pytorch_lightning.callbacks.EarlyStopping`。
- `model_summary.yaml`：实例化 `RichModelSummary`。
- `rich_progress_bar.yaml`：实例化 `RichProgressBar`。
- `learning_rate_monitor.yaml`：实例化 `LearningRateMonitor`。

如果做 overfit 调试，`debug/overfit.yaml` 会将 `callbacks: null`，避免 checkpoint 和 early stopping 干扰。

## 11. Logger 配置 `logger/`

### 11.1 `logger/tensorboard.yaml`

默认 logger，实例化：

```yaml
_target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger
```

输出目录：

```text
${paths.output_dir}/tensorboard/
```

### 11.2 `logger/csv.yaml`

实例化 Lightning `CSVLogger`：

```yaml
save_dir: ${paths.output_dir}
name: csv/
```

### 11.3 `logger/many_loggers.yaml`

同时启用 CSV 和 TensorBoard：

```yaml
defaults:
  - csv.yaml
  - tensorboard.yaml
```

使用方式：

```bash
python train.py experiment=final_version logger=many_loggers
```

## 12. Extras 配置 `extras/default.yaml`

由 `src.utils.extras(cfg)` 在任务开始前读取。

默认值：

- `ignore_warnings: True`：忽略 Python warnings。
- `enforce_tags: True`：没有 tags 时要求补充 tags，并保存。
- `print_config: True`：用 Rich 打印完整配置树并保存。

调试时如果不想交互或打印太多，可以覆盖：

```bash
python train.py experiment=final_version extras.enforce_tags=false extras.print_config=false
```

## 13. Debug 配置 `debug/`

Debug 配置都使用 `# @package _global_`，直接覆盖全局配置。

### 13.1 `debug/default.yaml`

基础调试配置：

- `mode: debug`：输出到 debug 目录。
- `extras.ignore_warnings: False`
- `extras.enforce_tags: False`
- Hydra job logging root level 改为 `DEBUG`。

文件里还保留了注释示例，可按需启用 CPU、单进程、anomaly detection、`num_workers=0` 等。

### 13.2 `debug/fdr.yaml`

fast dev run：

```yaml
trainer:
  fast_dev_run: true
```

用于只跑 1 个 train、1 个 val、1 个 test step。

### 13.3 `debug/limit.yaml`

限制数据量：

```yaml
trainer:
  max_epochs: 3
  limit_train_batches: 0.01
  limit_val_batches: 0.05
  limit_test_batches: 0.05
```

### 13.4 `debug/overfit.yaml`

过拟合少量 batch：

```yaml
trainer:
  max_epochs: 20
  overfit_batches: 3
callbacks: null
```

### 13.5 `debug/profiler.yaml`

开启 profiler：

```yaml
trainer:
  max_epochs: 1
  profiler: simple
```

可按注释改成 `advanced` 或 `pytorch`。

## 14. 配置覆盖关系与优先级

Hydra defaults 的顺序很重要。通常后面的配置会覆盖前面的同名字段。

训练时大致过程：

```text
train.yaml
  -> hparams/default.yaml
  -> data/nyu.yaml
  -> model/default.yaml
  -> callbacks/default.yaml
  -> logger/tensorboard.yaml
  -> trainer/ddp.yaml
  -> paths/default.yaml
  -> extras/default.yaml
  -> hydra/default.yaml
  -> experiment=<指定实验>
  -> debug=<可选调试配置>
  -> 命令行覆盖项
```

例如：

```bash
python train.py experiment=final_version_kitti trainer=gpu trainer.devices=1 batch_size=1
```

最终效果：

- `experiment=final_version_kitti` 把数据覆盖成 KITTI，把模型补齐成 final version 结构。
- `trainer=gpu` 把默认 DDP 改成单卡 GPU。
- `trainer.devices=1` 明确使用 1 张卡。
- `batch_size=1` 覆盖 experiment 中的 `batch_size: 2`。

## 15. 常用命令模板

NYU 训练：

```bash
python train.py experiment=final_version
```

KITTI 训练：

```bash
python train.py experiment=final_version_kitti
```

单卡调试：

```bash
python train.py experiment=final_version trainer=gpu trainer.devices=1
```

快速检查完整链路：

```bash
python train.py experiment=final_version trainer=gpu trainer.devices=1 debug=fdr
```

限制数据量调试：

```bash
python train.py experiment=final_version trainer=gpu trainer.devices=1 debug=limit
```

改数据根目录：

```bash
python train.py experiment=final_version paths.data_dir=/your/data/root
```

关闭 sparse 增强：

```bash
python train.py experiment=final_version is_sparse=false
```

使用 no fill 消融：

```bash
python train.py experiment=no_fill
```

KITTI select lines32 评测示例：

```bash
python eval.py experiment=final_version_kitti_test ckpt_path=pretrain/kitti.ckpt data.args.num_lines=lines32
```

## 16. 最容易踩的配置点

- 不要裸跑 `python train.py` 或 `python eval.py`，当前默认 `model: default` / `experiment: default` 在仓库里并不完整。
- `train.yaml` 默认 `trainer: ddp`，单卡或本地调试要覆盖 `trainer=gpu trainer.devices=1`。
- `paths.data_dir` 是绝对路径，换机器需要覆盖。
- KITTI 的 `val: full/select` 影响读取目录，评测 lines64/32/16/8/4 通常需要 `val: select`。
- KITTI `max_depth` 应为 `80.0`，NYU/SUNRGBD 为 `10.0`。
- 模型结构参数主要在 `experiment/*.yaml` 中，不要只看 `model/Uncertainty.yaml` 的基础默认值。
- `NLSPN` 依赖 DCN 编译，配置启用 refiner 时必须先编译 `src/plugins/deformconv`。
- `eval.py` 中 KITTI 走 `trainer.validate(...)`，不要按 NYU/SUNRGBD 的 `trainer.test(...)` 理解。
