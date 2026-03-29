# Notes

## TODOList

- 待补充

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
