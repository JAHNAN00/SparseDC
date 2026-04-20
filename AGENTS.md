# AGENTS 指南（SparseDC）

## 先看哪里（高优先级）
- 先读 `README.md`、`configs/`、`train.py`、`eval.py`，不要先从 `src/models/*` 叶子模块开始。
- 以可执行配置为准：`configs/*.yaml` 的事实优先于 README 文案。

## 一键命令（避免猜错）
- 环境：`mamba env create -f environment.yaml && mamba activate SparseDC`（执行任何训练/评测前先确认已激活）
- 依赖：`mim install mmcv-full`
- 编译 DCN（必需，`import DCN`）：`python src/plugins/deformconv/setup.py build install`
- 训练（NYU）：`python train.py experiment=final_version`
- 训练（KITTI）：`python train.py experiment=final_version_kitti`
- 评测（NYU 全套采样）：`./eval_nyu.sh final_version final_version pretrain/nyu.ckpt`
- 评测（KITTI lines64/32/16/8/4）：`./eval_kitti.sh final_version_kitti final_version_kitti_test pretrain/kitti.ckpt`
- 评测（SUNRGBD）：`./eval_sunrgbd.sh final_version final_version pretrain/nyu.ckpt`

## 关键坑点（最容易踩）
- 不要裸跑 `python train.py` / `python eval.py`：默认引用的 `model: default`、`experiment: default` 在仓库中不存在；请显式传 `experiment=...`。
- `train.yaml` 默认 `trainer: ddp`（`devices: -1`）；单卡调试请显式改为 `trainer=gpu trainer.devices=1`（或直接用 `CUDA_VISIBLE_DEVICES=0` 配合覆盖）。
- `eval.py` 对 `kitti` 走 `trainer.validate(...)`，对 `nyu/sunrgbd` 走 `trainer.test(...)`；不要按“统一 test 流程”改坏。
- 三个 `eval_*.sh` 都硬编码了 `CUDA_VISIBLE_DEVICES=0`，多卡/指定其他卡时要覆盖或改脚本。

## 数据与路径约束
- 数据根目录来自 `configs/paths/default.yaml` 的 `paths.data_dir`（当前是绝对路径 `/home/an/Desktop/SparseDC/data`）。换机器优先用命令行覆盖：`paths.data_dir=/your/data/root`。
- NYU 读取 `${paths.data_dir}/nyudepthv2`，KITTI 读取 `${paths.data_dir}/kitti_depth` + `${paths.data_dir}/kitti_raw`，SUNRGBD 读取 `${paths.data_dir}/SUNRGBD`。
- `data/` 下数据集为软链接，真实数据在项目目录外的机械硬盘；非必要禁止写入数据集内容。

## 输出与产物位置
- Hydra 输出目录：`logs/<data>/<model>/<mode>/<task_name>/<timestamp>/`（见 `configs/hydra/default.yaml`）。
- 训练/评测指标汇总默认在每次运行目录的 `val.csv`；逐样本结果常见为 `output.csv`。
- 代码会在输出目录创建可视化与测试结果子目录（如 `val_results/`、`test/`），不要误删后再报“结果缺失”。

## 改动建议（针对代理）
- 优先改 `configs/experiment/*.yaml` 做实验开关，不要先改模型代码硬编码超参。
- 若只做最小可运行验证，优先跑单个评测命令而非完整训练；此仓库无现成 lint/typecheck/pytest 流程可依赖。

## 操作安全与耗时约束
- 严禁执行会对远程仓库造成不可逆损失的操作（尤其是改写历史/不可恢复删除）；允许常规 `commit`/`push`，且 `commit message` 使用中文。
- 对可能运行超过 2 小时的命令，先给出耗时判断并二次确认后再执行。
