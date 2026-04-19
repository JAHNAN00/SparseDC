# SUNRGBD 训练测试记录

## 代码改动

- 修复 `src/data/sunrgbd.py` 中训练分支未生成 `dep_sp` 的问题。
- 让 `rgb`、`gt`、`depth input` 在训练增强时保持一致的翻转与旋转。
- 修正 `SUNRGBD` 训练/验证/测试目录选择逻辑。
- 新增 `configs/experiment/final_version_sunrgbd.yaml`，用于在 `SUNRGBD` 上从 `pvt_v2_b1.pth` 初始化训练。

## 资源选择

- GPU: `RTX 2080 Ti 11GB`
- 稳定训练配置:
  - `trainer=gpu`
  - `trainer.devices=1`
  - `+trainer.precision=16`
  - `batch_size=4`
  - `data.num_workers=4`
- 试探结论:
  - `batch_size=8` 会 OOM
  - `batch_size=4 + AMP` 稳定

## 关键命令

### 冒烟测试

```bash
python train.py experiment=final_version_sunrgbd trainer=gpu trainer.devices=1 ++trainer.precision=16 trainer.max_epochs=1 +trainer.limit_train_batches=2 +trainer.limit_val_batches=2 data.num_workers=4 task_name=sunrgbd_smoketest
```

### 正式训练第 1 段（0-9 epoch）

```bash
python train.py experiment=final_version_sunrgbd trainer=gpu trainer.devices=1 ++trainer.precision=16 batch_size=4 data.num_workers=4 trainer.max_epochs=10 test=False task_name=sunrgbd_train10
```

输出目录:

`logs/sunrgbd/Uncertainty/train/sunrgbd_train10/2026-04-20_00-03-05/`

### 正式训练第 2 段（续训到 19 epoch）

```bash
python train.py experiment=final_version_sunrgbd trainer=gpu trainer.devices=1 ++trainer.precision=16 batch_size=4 data.num_workers=4 trainer.max_epochs=20 test=False ckpt_path="/home/an/Desktop/SparseDC/logs/sunrgbd/Uncertainty/train/sunrgbd_train10/2026-04-20_00-03-05/checkpoints/last.ckpt" task_name=sunrgbd_train20
```

输出目录:

`logs/sunrgbd/Uncertainty/train/sunrgbd_train20/2026-04-20_01-31-05/`

### 正式测试

```bash
python eval.py experiment=final_version_sunrgbd data=sunrgbd ckpt_path="/home/an/Desktop/SparseDC/logs/sunrgbd/Uncertainty/train/sunrgbd_train20/2026-04-20_01-31-05/checkpoints/epoch_018.ckpt" trainer=gpu trainer.devices=1 task_name=sunrgbd_eval_from_sunrgbd_train20
```

输出目录:

`logs/sunrgbd/Uncertainty/eval/sunrgbd_eval_from_sunrgbd_train20/2026-04-20_03-00-05/`

## 训练结果

- `train10` 最优验证结果:
  - `RMSE = 0.17146` at epoch `8`
- `train20` 最优验证结果:
  - `RMSE = 0.16199` at epoch `18`

最佳 checkpoint:

`/home/an/Desktop/SparseDC/logs/sunrgbd/Uncertainty/train/sunrgbd_train20/2026-04-20_01-31-05/checkpoints/epoch_018.ckpt`

## 测试结果

来自:

`logs/sunrgbd/Uncertainty/eval/sunrgbd_eval_from_sunrgbd_train20/2026-04-20_03-00-05/val.csv`

```text
RMSE  = 0.10041
MAE   = 0.02701
iRMSE = 0.43170
iMAE  = 0.02782
REL   = 0.02293
D^1   = 0.98386
D^2   = 0.99177
D^3   = 0.99558
```

## 备注

- 全过程未改动原始数据集内容，仅读取 `data/SUNRGBD`。
- 远程推送当前仍被本机 Git HTTPS 认证拦截，需要本机具备 GitHub 凭证后再执行 `git push origin main`。
