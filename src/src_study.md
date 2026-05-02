# SparseDC src 源码学习笔记

本文基于当前仓库的 `src/` 源码、`train.py`、`eval.py` 与 `configs/` 配置整理，目标是快速理解项目构成、训练/评测执行逻辑和模型前向数据流。

## 1. 项目定位

SparseDC 是一个基于 PyTorch Lightning + Hydra 的深度补全项目。输入通常是 RGB 图像和稀疏深度 `dep`，输出稠密深度预测。核心模型围绕三部分展开：

- SFFM：用 RGB 特征辅助填充稀疏深度特征，对应 `FillConv` 与 `is_fill` 开关。
- 双分支特征提取：local CNN 分支 `ResNetU_` 与 global ViT 分支 `PVTV2`。
- UFFM + refine：`UncertaintyFuse_` 基于不确定度融合双分支预测，之后可接 `NLSPN` 做空间传播细化。

常用运行入口不在 `src/` 下，而是根目录的 `train.py` 和 `eval.py`。源码对象主要由 Hydra 根据 `configs/*.yaml` 的 `_target_` 实例化。

## 2. 顶层执行入口

### 2.1 训练入口 `train.py`

推荐命令：

```bash
python train.py experiment=final_version
python train.py experiment=final_version_kitti
```

执行顺序：

1. Hydra 读取 `configs/train.yaml`，再叠加 `configs/experiment/*.yaml`。
2. `fix_DictConfig(cfg)` 递归触发配置字段解析。
3. `@utils.task_wrapper` 包装 `train(cfg)`，执行额外配置打印、异常记录、logger 关闭等。
4. 如设置 `seed`，调用 `pl.seed_everything(cfg.seed, workers=True)`。
5. 通过 Hydra 实例化：
   - `cfg.data` -> `src.data.DataModule`
   - `cfg.model` -> `src.models.model.DepthLitModule`
   - `cfg.callbacks` -> Lightning callbacks
   - `cfg.logger` -> Lightning loggers
   - `cfg.trainer` -> Lightning `Trainer`
6. `trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))` 开始训练。
7. 若 `cfg.test=True`，使用 best checkpoint 执行 `trainer.test(...)`。

注意：不要裸跑 `python train.py`，因为默认 `model: default`、`experiment: default` 在当前仓库中并不完整。应显式指定 `experiment=final_version` 或其他存在的实验配置。

### 2.2 评测入口 `eval.py`

推荐脚本：

```bash
./eval_nyu.sh final_version final_version pretrain/nyu.ckpt
./eval_kitti.sh final_version_kitti final_version_kitti_test pretrain/kitti.ckpt
./eval_sunrgbd.sh final_version final_version pretrain/nyu.ckpt
```

执行顺序与训练类似，区别是必须提供 `ckpt_path`，且不同数据集分支不同：

- `nyu`：循环 10 次 `trainer.test(...)`，每次使用 `cfg.seed + i`，最后写均值和标准差。
- `sunrgbd`：执行 1 次 `trainer.test(...)`。
- `kitti`：执行 `trainer.validate(...)`，代码注释明确这是 KITTI test 的流程。

## 3. 配置如何连接源码

关键配置文件：

- `configs/train.yaml`：训练默认配置入口，默认数据是 `nyu`，默认 trainer 是 `ddp`，实验配置默认为 `null`。
- `configs/eval.yaml`：评测默认配置入口，默认 trainer 是 `gpu`，必须传 `ckpt_path`。
- `configs/hparams/default.yaml`：公共超参，如 `seed`、`monitor`、`num_sample`、`max_depth`、`is_sparse`。
- `configs/data/*.yaml`：选择 `DataModule` 和数据集参数。
- `configs/model/Uncertainty.yaml`：定义 LightningModule、网络、优化器、scheduler、metric。
- `configs/experiment/final_version*.yaml`：真正可运行的实验覆盖项。

以 `configs/experiment/final_version.yaml` 为例，核心 `_target_` 对应关系如下：

```text
model -> src.models.model.DepthLitModule
model.net -> src.models.base.Uncertainty_
model.net.backbone_l -> src.models.backbones.ResNetU_
model.net.backbone_g -> src.models.backbones.PVTV2
model.net.decode -> src.models.decodes.UncertaintyFuse_
model.net.criterion -> src.criterion.loss.DepthLoss
model.net.refiner -> src.models.refiners.NLSPN
model.metric -> src.criterion.metric.DepthCompletionMetric
data -> src.data.DataModule
```

NYU 使用 `max_depth=10.0`、`batch_size=8`。KITTI 使用 `max_depth=80.0`、`batch_size=2`，且 `model.net.is_padding=false`，输入尺寸按 KITTI 裁剪配置处理。

## 4. src 目录结构总览

```text
src/
├── criterion/      # loss 与 metric
├── data/           # DataModule、NYU/KITTI/SUNRGBD 数据集与数据增强
├── models/         # LightningModule、主网络、backbone、decode、refiner、模型工具
├── plugins/        # deformable convolution / DCN 插件，NLSPN 依赖
└── utils/          # 日志、分布式工具、可视化保存、Hydra/Lightning 辅助函数
```

## 5. 数据层 `src/data`

### 5.1 `datamodule.py`

`DepthDataModule` 是 Lightning DataModule。`src/data/__init__.py` 将其导出为 `DataModule`，所以配置中写的是：

```yaml
_target_: src.data.DataModule
```

初始化时根据 `dataset` 选择 Dataset 类：

- `kitti` -> `KittiDataset`
- `nyu` -> `NYUDataset`
- `sunrgbd` -> `SUNDataset`

`setup(stage)` 逻辑：

- `fit` 或 `validate`：构建 train 和 val dataset。
- `test`：构建 test dataset。

DataLoader 逻辑：

- train：使用配置中的 `batch_size`，`shuffle=True`。
- val：使用配置中的 `batch_size`，`shuffle=False`。
- test：固定 `batch_size=1`、`num_workers=1`。

### 5.2 `nyu.py`

`NYUDataset` 支持 `train/val/test` 三种模式。默认读取 `${paths.data_dir}/nyudepthv2/nyu.json` 中对应 split。

单样本输出：

```python
{"rgb": rgb, "dep": dep_sp, "gt": dep}
```

主要处理流程：

1. 从 `.h5` 或图片路径读取 RGB 和深度。
2. RGB 转 PIL，深度转 float32 PIL。
3. train 模式下可做随机缩放、旋转、水平翻转、颜色抖动、中心裁剪。
4. val/test 模式做固定 resize + center crop。
5. 根据 `num_sample` 生成稀疏深度 `dep_sp`。

稀疏采样支持多种模式：

- 整数：从有效深度点随机采样固定数量点。
- `shift_grid` / `shift_grid_N`：局部网格或局部随机采样。
- `uneven_density`：制造非均匀密度。
- `holes`：先采点，再挖掉局部区域。
- `keypoints_sift` / `keypoints_orb`：基于图像关键点采样。
- `short_range`：偏向较近深度。
- `up_fov`：偏向局部视场。

当 `mode=train` 且 `is_sparse=True` 时，会在 `5` 到 `num_sample` 之间随机采样点数，增强对稀疏输入的鲁棒性。

### 5.3 `kitti.py`

`KittiDataset` 的路径由 `get_paths_and_transform(split, args)` 生成。它会根据 split 和 `args.val` 选择不同目录：

- train：`data_depth_velodyne/train` 与 `data_depth_annotated/train`。
- val + `full`：KITTI 官方 train/val full 路径。
- val + `select`：`data_depth_selection/val_selection_cropped`，用于 lines64/32/16/8/4 等采样评测。
- test：anonymous test 路径。

单样本输出主要包括：

```python
{"rgb": rgb, "dep": sparse, "gt": target, "g": gray, "position": position}
```

其中 `g` 只有配置 `use_g=True` 时存在。模型主流程实际依赖 `rgb`、`dep`、`gt`。

KITTI 数据增强：

- train：底部裁剪到 `val_h/val_w`，随机水平翻转，RGB color jitter，再可随机 crop 到 `random_crop_height/random_crop_width`。
- val：底部裁剪。
- test：不做 transform。

当 `split=train` 且 `is_sparse=True` 时，会调用 `random_sparse`，按随机比例保留稀疏深度行/点。

### 5.4 `sunrgbd.py`

`SUNDataset` 读取 SUNRGBD 组织后的图片和深度：

- train/val：从 `train_depth` 或 fallback `train_depth_gt` 获取文件名。
- test：从 `test_depth_gt` 或 fallback `test_depth` 获取文件名。
- train/val 通过 `radio` 切分，默认 `radio=0.2` 作为验证比例。

单样本输出：

```python
{"rgb": rgb, "dep": dep_sp, "gt": dep}
```

深度读取后除以 `10000.0`，并使用 `AdaResize` 对稀疏深度做带 mask 的插值，避免直接 resize 稀疏图导致数值污染。

### 5.5 `transforms.py` 与 `CoordConv.py`

`transforms.py` 是 KITTI 数据增强工具集合，包含 `Compose`、`ToTensor`、`Rotate`、`Resize`、`BottomCrop`、`RandomCrop`、`HorizontalFlip`、`ColorJitter`、`CutFlip` 等。

`CoordConv.py` 用于生成坐标通道，KITTI Dataset 中生成 `position` 字段，但当前核心模型前向主要使用 `rgb/dep/gt`。

## 6. Lightning 模型封装 `src/models/model.py`

`DepthLitModule` 是训练和评测循环的中心封装，不直接定义网络结构，而是持有 `self.net`。`self.net` 通常是 `Uncertainty_`。

### 6.1 初始化

构造参数来自 `configs/model/Uncertainty.yaml`：

- `net`：真正的深度补全网络。
- `optimizer`：默认 Adam partial。
- `scheduler`：默认 ReduceLROnPlateau partial。
- `metric`：`DepthCompletionMetric`。
- `monitor`：默认 `RMSE`。
- `save_dir`：Hydra 输出目录。
- `dataset`：当前数据集名。
- `is_warmup`：是否做一个 epoch 内的线性 warmup。

rank0 会创建：

- `val.csv`：每个 epoch 的验证/测试汇总。
- `val_results/`：验证可视化图片。

### 6.2 训练逻辑

`training_step`：

```python
_, loss, loss_val = self.forward(batch)
```

模型返回预测、总 loss、loss 字典。训练阶段主要记录平均 loss 与各子 loss。

`optimizer_step`：

- 正常 `optimizer.step(...)`。
- 若 `is_warmup=True`，在第一个训练 epoch 的 batch 维度线性增加学习率到 `base_lr`。

### 6.3 验证逻辑

`validation_step`：

1. 前向得到 `pred`。
2. 使用 `DepthCompletionMetric.evaluate(pred, gt)` 计算当前 batch 指标。
3. 将逐样本指标写入 `output.csv`。
4. 第一个 batch 保存可视化结果到 `val_results/epoch_*.png`。

`validation_epoch_end`：

1. 调用 `metric.average()` 得到平均指标。
2. 分布式情况下通过 `reduce_value` 汇总。
3. 记录 `val/RMSE` 等指标。
4. 更新 `val/best_result`。
5. 将本 epoch 平均指标写入 `val.csv`。

### 6.4 测试逻辑

`on_test_start` 创建 `test/` 目录和测试 CSV。

`test_step`：

- NYU/SUNRGBD：计算指标、写 CSV、保存拼接可视化图。
- KITTI：保存预测深度为 uint16 PNG，用于上传或评测。

`test_epoch_end` 只对 NYU/SUNRGBD 聚合指标并写入 `val.csv`。KITTI 的 `eval.py` 实际走 `validate`，不是这里的 `test_epoch_end`。

## 7. 主网络 `src/models/base/uncertainty.py`

`Uncertainty_` 是 SparseDC 的核心网络组合器。

### 7.1 子模块

常见配置下包含：

- `backbone_l`: `ResNetU_`，局部 CNN 分支。
- `backbone_g`: `PVTV2`，全局 Transformer 分支。
- `decode`: `UncertaintyFuse_`，双分支不确定度融合解码器。
- `criterion`: `DepthLoss`，最终 refine 输出的监督 loss。
- `refiner`: `NLSPN`，可选空间传播细化模块。
- `fill_conv`: `FillConv`，当 `is_fill=True` 时启用，用 RGB 特征稳定稀疏深度特征。

### 7.2 前向输入输出

输入 batch 至少包含：

```python
sample["rgb"]  # B x 3 x H x W
sample["dep"]  # B x 1 x H x W，稀疏深度
sample["gt"]   # B x 1 x H x W，训练/验证真值
```

输出：

```python
depth, loss, loss_val
```

### 7.3 前向执行流程

完整流程：

1. 选择监督目标：训练时 `gt=sample["gt"]`，非训练时当前代码先设 `gt=sample["dep"]`，后续验证/测试外部 metric 仍用 batch 的 `gt`。
2. 若 `is_padding=True`，对 `rgb` 和 `dep` 做中心 padding 到 `padding_size`。
3. 将 `dep` 与 `gt` clamp 到 `[0, max_depth]`。
4. 若 `is_fill=True`：
   - `f = FillConv(rgb, dep)` 生成 64 通道融合特征。
   - `f_depth = d_conv(f)` 生成一个辅助原始深度预测。
   - local/global backbone 都使用 `(rgb, dep, f)`。
5. 若 `is_fill=False`：
   - local/global backbone 自己拼接 RGB 与深度浅层特征。
6. `decode(outs, dep)` 返回：
   - `x`：融合特征。
   - `depths`：多尺度 local/global/fuse 深度预测。
   - `uncertainties`：多尺度 local/global/fuse 不确定度预测。
7. `get_loss(gt, depths, uncertainties, dep)` 对多尺度预测做深度和不确定度监督。
8. 若 `is_fill=True`，额外加入 `f_depth` 辅助 loss，权重为 `0.05`。
9. 取最终融合深度 `depths["fuse_d"][-1]` 作为 `pred_init`，置信度 `conf = 1 - uncertainties["fuse_u"][-1]`。
10. 若启用 `refiner`：
    - `guide = guide_layer(x)` 生成 NLSPN guidance。
    - `NLSPN(pred_init, guide, conf, dep, rgb)` 传播细化。
    - 对 refine 输出计算 `DepthLoss`。
11. 返回 refine 后深度；若无 refiner，则返回初始融合深度。

### 7.4 多尺度 loss

`get_loss` 会对 `depths` 和 `uncertainties` 的多尺度输出从粗到细计算 loss：

- 深度 loss：`((d - dep) ** 2)[m].mean()`。
- 不确定度目标：`compute_uncertainty(d, dep, ratio)`。
- 分支：`local`、`global`、`fuse` 都参与。
- 权重：第 `i` 层使用 `0.8 ** i` 衰减。

其中 `adapt_pool` 用 mask-aware average pooling 将监督深度下采样到下一尺度。

## 8. Backbone

### 8.1 `src/models/backbones/resnet.py`

主要使用 `ResNetU_`。

`ResNetU_` 逻辑：

- `is_fill=True`：直接接收外部传入的 64 通道填充特征 `f`，再过一层 `conv`。
- `is_fill=False`：分别用 `conv_rgb(3->48)` 和 `conv_d(1->16)` 提取浅层特征，再 concat 成 64 通道。
- 主干使用 torchvision 的 ResNet layer1-layer4，再额外加 `conv5`，输出 5 个尺度特征。

常用 `resnet18` 的 `num_features` 为：

```python
[64, 128, 256, 512, 512]
```

### 8.2 `src/models/backbones/pvtv2.py`

`PVTV2` 是 global 分支封装。

`PVTV2` 逻辑：

- `is_fill=True`：接收外部 64 通道 `f`。
- `is_fill=False`：与 `ResNetU_` 类似，先把 RGB 和深度浅层特征拼成 64 通道。
- 内部 `PyramidVisionTransformerV2` 以 64 通道作为输入，输出 4 个尺度的全局特征。
- 可通过 `pretrained` 加载 PVT 预训练权重，常见路径是 `pretrain/pvt_v2_b1.pth`。

`pvt_v2_b1` 的输出通道配置为：

```python
[64, 128, 320, 512]
```

## 9. 解码融合 `src/models/decodes/uncertainty.py`

核心类是 `UncertaintyFuse_`。它负责把 local CNN 特征和 global PVT 特征自顶向下融合，并同时预测深度与不确定度。

### 9.1 重要组件

- `UDConv`：从特征中预测深度 `d` 和不确定度 `u`。非首层会基于上一层 `d/u` 做残差更新。
- `FuseConv`：融合 local/global 特征。若 `is_gate_fuse=True`，会使用不确定度作为门控权重。
- `GateConv`：门控卷积块。
- `get_depth_pool`：对输入稀疏深度做多尺度 mask-aware pooling，为解码过程提供不同尺度的 sparse depth 和 mask。

### 9.2 解码流程

`forward(outs, depth)` 中：

1. 输入 `outs=(local_f, global_f)`。
2. 将 local/global 特征列表反转，从最粗尺度开始自顶向下解码。
3. 对输入稀疏深度生成多尺度 `masks` 和 `sparse_d`，也反转到粗到细顺序。
4. 初始阶段只建立 global/local 的粗尺度特征。
5. 从可预测尺度开始，同时更新：
   - global 深度与不确定度：`g_d/g_u`
   - local 深度与不确定度：`l_d/l_u`
   - fuse 深度与不确定度：`d/u`
6. 在有原始稀疏深度的位置，使用观测深度替换或约束预测：
   - `d = d * ~_m + _d * _m`
   - `u = u * ~_m + 0.05 * _m`
7. 每层输出追加到：
   - `depths["local_d"]`
   - `depths["global_d"]`
   - `depths["fuse_d"]`
   - `uncertainties["local_u"]`
   - `uncertainties["global_u"]`
   - `uncertainties["fuse_u"]`
8. 返回最终融合特征 `f_prev`、多尺度深度字典和不确定度字典。

这部分就是论文中 UFFM 思想在代码中的主要落点：用 local/global 分支各自预测的不确定度来指导融合。

## 10. Refiner `src/models/refiners/NLSPN.py`

`NLSPN` 是非局部空间传播网络，来自 NLSPN ECCV 2020。它依赖 `src/plugins` 中的 modulated deform conv。

输入：

```python
feat_init   # 初始深度预测，B x 1 x H x W
guidance    # guide_layer 生成的传播引导，B x ch_g x H x W
confidence  # 通常是 1 - uncertainty
feat_fix    # 原始稀疏深度 dep
rgb         # RGB，可选传入
```

关键流程：

1. `conv_offset_aff(guidance)` 预测 deform conv offset 和 affinity。
2. 若 `conf_prop=True`，用 confidence 调制 affinity。
3. 对 affinity 做归一化，并补上中心点 affinity。
4. 重复 `prop_time` 次调用 `ModulatedDeformConvFunction.apply` 传播深度。
5. 返回传播后的深度。

由于 NLSPN 依赖自定义 DCN，运行前需要编译：

```bash
python src/plugins/deformconv/setup.py build install
```

## 11. Loss 与 Metric `src/criterion`

### 11.1 `loss.py`

`DepthLoss` 继承 `BaseLoss`，通过字符串配置组合多个 loss。例如：

```yaml
loss: 1.0*L2+1.0*L1
```

会实例化：

- `L2Loss`
- `L1Loss`

可用 loss 包括：

- `L1Loss`：有效深度区域的绝对误差。
- `L2Loss`：有效深度区域的平方误差。
- `SiLogLoss`：尺度不变 log loss。
- `EdgeLoss`：基于深度梯度的平滑/边缘项。
- `GradLoss`：Sobel 梯度差异。

有效区域通常是 `gt > 0.0001`，并会 clamp 到 `[0, max_depth]`。

### 11.2 `metric.py`

`DepthCompletionMetric` 计算深度补全常用指标：

- `RMSE`
- `MAE`
- `iRMSE`
- `iMAE`
- `REL`
- `D^1`
- `D^2`
- `D^3`

`evaluate(pred, gt)` 计算当前 batch 指标并累计，`average()` 返回按样本数加权的平均值，`reset()` 清空状态。

## 12. 工具层 `src/utils`

### 12.1 `utils.py`

核心函数：

- `task_wrapper`：包装 train/eval 主函数，负责执行 extras、异常记录、输出目录打印、关闭 logger。
- `extras`：根据配置处理 warnings、tags、Rich config tree。
- `instantiate_callbacks`：根据 Hydra 配置实例化 callbacks。
- `instantiate_loggers`：根据 Hydra 配置实例化 loggers。
- `log_hyperparameters`：记录模型参数量、配置、trainer/callback/logger 信息。
- `close_loggers`：关闭 wandb 等 logger。

### 12.2 `dist_utils.py`

提供分布式辅助：

- `get_dist_info()`：获取当前 rank 和 world size。
- `is_master()`：判断是否 rank0。
- `reduce_value()`：多卡下 reduce 指标。

### 12.3 `vis_utils.py`

负责保存验证和测试可视化：

- `batch_save`：NYU/SUNRGBD 可视化。
- `batch_save_kitti`：KITTI 可视化。
- `merge_into_row`：拼接 RGB、稀疏深度、预测、GT、误差图。
- `save_depth_as_uint16png_upload`：KITTI 输出 uint16 深度 PNG。

## 13. 插件层 `src/plugins`

`src/plugins` 提供 deformable convolution 相关实现。`NLSPN` 直接 import：

```python
from src.plugins import ModulatedDeformConvFunction
```

相关文件：

- `src/plugins/modulated_deform_conv_func.py`
- `src/plugins/deformconv/functions/*.py`
- `src/plugins/deformconv/modules/*.py`
- `src/plugins/deformconv/setup.py`

这部分不是普通 Python 纯源码，依赖 C/CUDA 扩展。未编译时，带 `NLSPN` 的模型会在 import 或 forward 阶段失败。

## 14. 一次训练 batch 的完整数据流

以下是 `python train.py experiment=final_version` 下的典型链路：

```text
Hydra compose configs
  -> instantiate DataModule / DepthLitModule / Trainer
  -> Trainer.fit
  -> DataModule.setup("fit")
  -> NYUDataset.__getitem__
  -> batch = {rgb, dep, gt}
  -> DepthLitModule.training_step(batch)
  -> DepthLitModule.forward(batch)
  -> Uncertainty_.forward(batch)
  -> optional FillConv(rgb, dep)
  -> ResNetU_(rgb, dep, f) + PVTV2(rgb, dep, f)
  -> UncertaintyFuse_(local/global features, dep)
  -> multi-scale depth/uncertainty losses
  -> optional NLSPN refine
  -> DepthLoss(refined_depth, gt)
  -> return depth, loss, loss_val
  -> Lightning backward / optimizer / scheduler
```

## 15. 一次验证/评测 batch 的完整数据流

验证链路：

```text
Trainer.validate or validation during fit
  -> DataModule.setup("validate")
  -> Dataset.__getitem__
  -> DepthLitModule.validation_step(batch)
  -> Uncertainty_.forward(batch)
  -> pred
  -> DepthCompletionMetric.evaluate(pred, batch["gt"])
  -> write output.csv
  -> save first-batch visualization
  -> validation_epoch_end aggregates metrics
  -> write val.csv
```

测试链路：

```text
Trainer.test
  -> DataModule.setup("test")
  -> DepthLitModule.test_step(batch)
  -> Uncertainty_.forward(batch)
  -> NYU/SUNRGBD: compute metrics + save visualization
  -> KITTI: save uint16 depth png
```

需要特别注意：`eval.py` 对 KITTI 使用 `trainer.validate(...)`，不是 `trainer.test(...)`。

## 16. 输出文件位置

Hydra 输出目录由 `configs/hydra/default.yaml` 控制，整体位于 `logs/...` 下。一次运行中常见文件：

- `val.csv`：epoch 级验证/测试汇总。
- `output.csv`：验证阶段逐样本指标。
- `val_results/`：验证可视化图。
- `test/`：测试输出，NYU/SUNRGBD 是可视化图与 CSV，KITTI 是深度 PNG。
- Lightning checkpoint：由 callbacks 配置控制。

## 17. 阅读源码建议

建议按以下顺序读源码：

1. `train.py`、`eval.py`：先理解 Hydra/Lightning 如何启动。
2. `configs/experiment/final_version*.yaml`：确认实际实例化的类和关键超参。
3. `src/data/datamodule.py` 与对应 dataset：理解 batch 字段。
4. `src/models/model.py`：理解 Lightning 训练、验证、测试步骤。
5. `src/models/base/uncertainty.py`：理解主网络如何串起所有子模块。
6. `src/models/backbones/resnet.py` 与 `src/models/backbones/pvtv2.py`：理解 local/global 特征来源。
7. `src/models/decodes/uncertainty.py`：重点理解多尺度不确定度融合。
8. `src/models/refiners/NLSPN.py`：理解 refine 和 DCN 依赖。
9. `src/criterion/loss.py` 与 `src/criterion/metric.py`：理解监督与评测口径。

## 18. 常见修改入口

- 改数据路径：优先命令行覆盖 `paths.data_dir=/your/data/root`。
- 改采样点数：覆盖 `num_sample=...`。
- 开关稀疏增强：覆盖 `is_sparse=true/false`。
- 修改模型结构：优先改 `configs/experiment/*.yaml` 中的 `model.net.*`。
- 去掉 fill：参考 `configs/experiment/no_fill.yaml` 或将 `is_fill=false` 并同步 backbone 配置。
- 去掉 sparse 训练增强：参考 `no_sparse` 配置。
- 修改 loss：改 `model.net.criterion.args.loss`，例如 `1.0*L2+1.0*L1`。
- 修改 refine 次数：改 `model.net.refiner.args.prop_time`。

## 19. 关键注意点

- `configs/paths/default.yaml` 当前数据根目录是绝对路径 `/home/an/Desktop/SparseDC/data`，换机器时应覆盖。
- 默认 trainer 是 DDP，单卡调试建议显式覆盖 `trainer=gpu trainer.devices=1`。
- `eval_*.sh` 默认写死 `CUDA_VISIBLE_DEVICES=0`。
- `NLSPN` 依赖 DCN 编译，未编译会失败。
- KITTI 的评测路径和 `val: full/select`、`num_lines` 强相关，读配置比读 README 更可靠。
- `DepthLitModule.test_step` 对 KITTI 会保存 PNG，但当前 `eval.py` 的 KITTI 分支走 validate，这是仓库设计，不要轻易改成统一 test。
