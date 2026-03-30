# Notes

## TODOList

- 等待并整理完整 `KITTI` 测试结果：
  - 跑完 `eval_kitti.sh` 的 `lines64/32/16/8/4`
  - 对照论文 Table 2 逐项核验
- 继续排查 `sunrgbd` 正常评测结果为何稳定低于论文：
  - 当前 `organized data` 是否与作者发布版本逐文件一致
  - `pretrain/nyu.ckpt` 是否确实对应论文 SUNRGBD 泛化实验所用权重
  - 是否还存在未对齐的评测前处理细节
- 后续处理 `nyuv2`：
  - 检查 `/media/an/4T/datasets/nyudepthv2` 中原始 `.mat` 数据的实际结构
  - 判断如何转换为项目期望的 `h5 + nyu.json` 格式

## 当前项目文件夹结构

```text
SparseDC
├── configs/                  配置文件
├── src/                      核心源码
├── media/                    README 配图等资源
├── pretrain/                 预训练模型目录
├── data/                     本地数据目录
│   ├── nyudepthv2/
│   ├── kitti_depth/
│   ├── kitti_raw/
│   └── SUNRGBD -> /media/an/4T/datasets/SUNRGBD
├── logs/                     训练与评估日志目录
├── analyse/                  分析输出目录
├── .vscode/                  编辑器配置
└── note.md                   当前笔记
```

### 说明

- `data/`、`pretrain/`、`logs/`、`analyse/` 都属于当前项目里常用但默认被 `.gitignore` 忽略的目录。
- `data/SUNRGBD` 当前是一个软链接，指向外部硬盘中的真实数据目录。
- `data/nyudepthv2` 当前是一个软链接，指向外部硬盘中的真实数据目录：`/media/an/4T/datasets/nyudepthv2`。
- `pretrain/` 用于放作者提供的 `.ckpt` 和 `pvt_v2_b1.pth` 等模型文件。

## 2026-03-29

### DCN 模块安装失败的直接原因

- 当前 `DCN` 没有安装成功，直接原因是 CUDA 编译工具链不兼容。
- 当前环境是 `PyTorch 1.12.1 + cu113`，系统里安装的是 `nvcc 11.5`。
- 编译 `DCN` 时，`nvcc` 在处理 `/usr/include/c++/11/bits/std_function.h` 时失败，报错为：
  `parameter packs not expanded with '...'`
- 这说明当前这套组合下，`nvcc 11.5` 与系统 `gcc/g++ 11` 以及当前 PyTorch 环境之间存在兼容性问题。

### DCN 安装的核心解决步骤

- 将编译器切换为 `gcc-10 / g++-10`。
- 保持 PyTorch 环境为 `torch 1.12.1 + cu113`。
- 编译时为 RTX 2080 Ti 指定架构：`TORCH_CUDA_ARCH_LIST=7.5`。
- 成功编译并安装 `DCN` 后，发现运行时还存在动态库路径问题。
- 通过给 `sparsedc` 环境添加 `conda activate/deactivate` 脚本，自动把 `torch/lib` 加入 `LD_LIBRARY_PATH`，最终解决 `import DCN` 时的 `libc10.so` 找不到的问题。

### 当前状态

- `DCN` 已经可以在 `sparsedc` 环境中正常导入。

## 2026-03-30

### SUNRGBD 测试排查

- 补充了作者提供的 `pretrain/nyu.ckpt`，并确认 SUNRGBD 测试应使用 `nyu.ckpt`，不是 `kitti.ckpt`。
- 将默认数据目录改为当前仓库本地路径：
  `configs/paths/default.yaml -> /home/an/Desktop/SparseDC/data`
- 确认当前 `data/SUNRGBD` 为软链接，目标目录存在，且包含：
  - `test_images`
  - `test_depth_gt`
  - `test_depth_input`
  - `train_images`
  - `train_depth_gt`
  - `train_depth_input`
- 确认当前 `organized data` 中没有作者原始代码期望的：
  - `test_depth`
  - `train_depth`

### 代码最小修改

- 修改 `src/data/datamodule.py`：
  - `setup(stage="test")` 时只初始化 `test` 数据集
  - 避免测试时因为缺少 `train/val` 目录而直接报错
- 修改 `src/data/sunrgbd.py`：
  - 增加最小兼容逻辑，优先读取作者原本目录名
  - 若不存在，则回退到当前 `organized data` 中的 `*_depth_gt` / `*_depth_input`

### 结果记录

- 论文 Table 3 中 SUNRGBD 的 `Ours`：
  - `RMSE = 0.1930`
  - `REL = 0.0710`
- 本地正常评测两次结果一致：
  - `RMSE = 0.2238`
  - `REL = 0.1103`
- 说明当前结果仍明显差于论文，且差距不是偶然波动。
- 已确认：
  - 当前 `input/gt` 语义与论文描述基本对得上
  - 论文与 README 都说明 SUNRGBD 测试应使用在 NYU 上训练的模型
- 当前怀疑点：
  - 作者公开的 `organized data` / `nyu.ckpt` 与论文实验资源之间可能存在未说明差异
  - 或评测流水线中仍有未对齐的细节

### 交换 `input/gt` 实验

- 做过一组额外实验：在 SUNRGBD 测试阶段交换 `input` 与 `gt`
- 实验目的：
  - 不是正式评测
  - 只用于排查作者是否可能在这里做了 hidden trick
- 实验日志：
  - 正常评测结果日志：
    `/home/an/Desktop/SparseDC/logs/sunrgbd/Uncertainty/eval/final_version/2026-03-30_17-30-09/val.csv`
  - 再次正常评测结果日志：
    `/home/an/Desktop/SparseDC/logs/sunrgbd/Uncertainty/eval/final_version/2026-03-30_18-03-15/val.csv`
  - 交换 `input/gt` 实验日志：
    `/home/an/Desktop/SparseDC/logs/sunrgbd/Uncertainty/eval/final_version_swap_io/2026-03-30_18-27-34/val.csv`
- 实验结果：
  - 正常评测：
    - `RMSE = 0.2238`
    - `REL = 0.1103`
  - 交换 `input/gt`：
    - `RMSE = 0.1061`
    - `REL = 0.0616`
  - 论文 Table 3：
    - `RMSE = 0.1930`
    - `REL = 0.0710`
- 结论：
  - 交换 `input/gt` 后的结果没有“撞上”论文数值，而是明显优于论文
  - 这说明该 benchmark 对 `input/gt` 语义非常敏感
  - 但这不能作为作者做了 hidden trick 的证据
  - 原因是交换后评测协议已经被破坏，模型输入和评测真值的语义发生了变化，结果不再可与论文正式对比
- 后续处理：
  - 已删除交换 `input/gt` 的实验代码
  - 当前代码恢复为正常可跑的 SUNRGBD 评测模式

### 版本记录

- 已提交一个可跑 SUNRGBD 的版本：
  - commit: `f45c7fd`
  - message: `Enable SUNRGBD evaluation with local organized data`

### KITTI 测试打通

- 继续整理 `KITTI` 数据：
  - 将 `data/kitti_depth` 指向本地准备好的 `kitti_depth`
  - 将 `data/kitti_raw` 指向本地准备好的 `kitti_raw`
  - 将 `data_depth_selection` 调整为项目代码实际期望的目录层级
- 修正 `configs/data/kitti.yaml`：
  - `data_folder` 改为 `/home/an/Desktop/SparseDC/data/kitti_depth`
  - 避免代码继续错误地去找多余的 `kitti_depth/depth`
- 修正 `src/data/datamodule.py`：
  - 将 `setup(stage="validate")` 重新纳入 `train/val` 数据初始化
  - 这是之前为 `SUNRGBD` 做“仅在 test 阶段初始化 test 数据集”改动后带来的副作用
  - `KITTI` 评测走的是 `trainer.validate()`，如果不补回 `validate`，会因为 `data_val` 为空而报：
    `TypeError: object of type 'NoneType' has no len()`
- `pvt_v2_b1.pth` 的加载 mismatch 只是预训练骨干与当前改造结构不完全匹配的提示：
  - 在 `KITTI` smoke test 中通过 `model.net.backbone_g.pretrained=null` 关闭该预训练加载
  - 真正评测仍由 `pretrain/kitti.ckpt` 恢复完整模型权重

### KITTI lines64 最小验证

- 运行命令：
  `CUDA_VISIBLE_DEVICES=0 python eval.py experiment=final_version_kitti_test ckpt_path=pretrain/kitti.ckpt task_name=final_version_kitti_lines64 ++data.args.num_lines=lines64 model.net.backbone_g.pretrained=null`
- 关键日志：
  - 配置日志：
    `/home/an/Desktop/SparseDC/logs/kitti/Uncertainty/eval/final_version_kitti_lines64/2026-03-30_20-58-07/config_tree.log`
  - 逐样本输出：
    `/home/an/Desktop/SparseDC/logs/kitti/Uncertainty/eval/final_version_kitti_lines64/2026-03-30_20-58-07/output.csv`
  - 汇总结果：
    `/home/an/Desktop/SparseDC/logs/kitti/Uncertainty/eval/final_version_kitti_lines64/2026-03-30_20-58-07/val.csv`
- 本地结果：
  - `RMSE = 0.79661`
  - `MAE = 0.20514`
- 论文 Table 2 中 `KITTI 64 lines / Ours`：
  - `RMSE = 0.7966`
  - `MAE = 0.2051`
- 结论：
  - 当前本地 `lines64` 最小验证结果与论文数值一致，可认为已成功复现该设置

### KITTI 可运行版本

- 当前可跑 `KITTI` 的代码结点：
  - commit: `fe50178`
  - message: `Fix KITTI eval after prior SUNRGBD datamodule change`
