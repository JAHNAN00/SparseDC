# Notes

## TODOList

- 补充当前机器上的训练/验证耗时基线：
  - `NYU` 训练：待补
  - `NYU` 验证：待补
  - `KITTI` 训练：待补
- 继续排查 `sunrgbd` 正常评测结果为何稳定低于论文：
  - 当前 `organized data` 是否与作者发布版本逐文件一致
  - `pretrain/nyu.ckpt` 是否确实对应论文 SUNRGBD 泛化实验所用权重
  - 是否还存在未对齐的评测前处理细节
- 后续处理 `nyuv2`：
  - 继续寻找作者 README 对应的 `sparse-to-dense` 预处理 NYUv2 数据
  - 补齐官方 `val/official` 与可直接使用的默认 `nyu.json` 所需 `.h5`

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

## 2026-03-31

### 当前机器上的大致耗时

- 统计口径：
  - 以日志目录时间戳作为启动时间
  - 以对应 `val.csv` 的落盘时间作为结束时间
  - 仅作为当前机器配置上的粗略墙钟时间参考
- `SUNRGBD` 正常验证：
  - `2026-03-30_17-30-09 -> 17:48:38`，约 `18 分 29 秒`
  - `2026-03-30_18-03-15 -> 18:19:50`，约 `16 分 36 秒`
  - 可粗略记为：`单次验证约 17~19 分钟`
- `KITTI` 完整验证：
  - `lines64`：`21:13:03 -> 21:18:06`，约 `5 分 04 秒`
  - `lines32`：`21:18:09 -> 21:23:21`，约 `5 分 13 秒`
  - `lines16`：`21:23:25 -> 21:28:19`，约 `4 分 55 秒`
  - `lines8`：`21:28:22 -> 21:33:13`，约 `4 分 51 秒`
  - `lines4`：`21:33:16 -> 21:38:08`，约 `4 分 53 秒`
  - 跑完 `eval_kitti.sh` 的 5 组设置总耗时约 `25 分钟`
- 预留占位：
  - `NYU` 训练：待补
  - `NYU` 验证：待补
  - `KITTI` 训练：待补

### KITTI 训练显存观察

- 当前机器上的 `KITTI` 训练尝试：
  - 默认 `FP32`、`batch_size=2` 会直接 OOM
  - 将 `batch_size` 降到 `1` 后，`FP32` 仍然 OOM
  - 典型报错显示：
    - GPU 总显存约 `10.57 GiB`
    - OOM 时 PyTorch 已分配约 `8.5~8.7 GiB`
    - 仅剩几十 MiB 可用显存
- 一个重要结论：
  - 在当前这张约 `11GB` 显存的卡上，`KITTI` 训练默认配置需要 `FP16` 才能把训练流程跑起来
  - 使用 `+trainer.precision=16` 后，训练可以启动
  - 但训练数据仍未补全，因此后续仍会因缺失 `kitti_raw` 中的部分 drive 而中断
- 对 `FP32` 训练显存需求的粗略估计：
  - 当前 `10.57 GiB` 显存下，`batch_size=1` 仍然不够
  - 结合 `FP16` 可启动、`FP32` 必炸这一现象，保守估计单卡 `FP32` 训练至少需要 `14~16 GiB`
  - 如果希望更稳，避免碎片与波动带来的边缘 OOM，`16GB+` 更合适
  - 因此 `RTX 4090 24GB` 从显存角度看应当足够支撑当前配置下的单卡训练

### KITTI raw 缺失包补下载

- 训练阶段报错定位到 `kitti_raw` 不完整，而不是 `kitti_depth` 本身有问题。
- 之前手头的下载清单 `data/kitti_archives_to_download.txt` 只覆盖了评测和部分 raw drive，不足以支撑完整训练。
- 已根据代码实际训练路径重新计算：
  - 以 `data_depth_annotated/train/*_sync` 与 `val/*_sync` 为准
  - 对照原始下载清单，生成新的补下载列表：
    `data/kitti_archives_needed_but_not_in_manifest.txt`
- 统计结果：
  - 当前代码训练/验证实际需要的 raw drive 并集共 `151` 个
  - 原始清单缺少其中 `89` 个 zip
  - 缺失项几乎都集中在 `2011_09_28`
- 当前进展：
  - 已确认并补解压 `2011_09_28_drive_0146_sync.zip`
  - 缺失的 raw 压缩包现已全部下载并解压完成
  - 按 `src/data/kitti.py` 训练时的真实路径规则复核后：
    - `data_depth_annotated/train`: `85898` 张
    - `data_depth_annotated/val`: `6852` 张
    - `data_depth_velodyne/train`: `85898` 张
    - `data_depth_velodyne/val`: `6852` 张
    - 训练/验证实际需要的 raw drive 共 `151` 个
    - 缺失 RGB 文件数：`0`
    - 缺失 drive 数：`0`
- 结论：
  - 当前 `KITTI` 数据问题已经解决，已满足这份代码的训练与评测数据要求
  - 当前主要瓶颈已从“缺失数据”转为“单卡显存限制下只能使用 `FP16 + batch_size=1`，训练耗时较长”

### NYUv2 `.mat` 提取排查

- 已确认 `/media/an/4T/datasets/nyudepthv2/nyu_depth_v2_labeled.mat` 可以被 `h5py` 正常读取。
- 其中包含的关键字段如：
  - `images`
  - `depths`
  - `rawRgbFilenames`
  - `scenes`
- 已编写并验证一个外部脚本：
  - `/media/an/4T/datasets/nyudepthv2/extract_nyuv2_from_mat.py`
  - 能将 `labeled.mat` 提取为项目可读的 `.h5` 样本，并生成 sidecar manifest
- 本地已完成 smoke test：
  - 从提取结果中切出 `50` 个样本作为临时 `val/test`
  - 评测命令可跑通，说明 `.mat -> .h5 -> loader -> eval` 链路本身没有问题
- 但需要明确：
  - `labeled.mat` 总共只有 `1449` 帧
  - 这不能替代作者 README 中使用的 `sparse-to-dense` 预处理 NYUv2 数据
  - 当前目录中的默认 `nyu.json` 对应的是更大规模的 `.h5` 数据集，而不是这份 `labeled.mat`
  - 因此直接使用默认 `nyu.json` 会因为大量 `.h5` 不存在而报错
- 这次 `smoke50` 结果不能与论文做严格复现对比：
  - 当前使用的是从 `train` 中临时切出的 `50` 个样本
  - 不是真正的官方 NYU test split
  - 只能用于验证流程是否跑通，不能用于判断是否复现论文 NYU 主结果
- 当前决定：
  - 将 `src/data/nyu.py` 恢复为作者原始逻辑，只读取默认 `nyu.json`
  - 等后续补到 README 对应的 NYUv2 预处理数据后，再继续正式 NYU 训练与评测
