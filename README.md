# Tiny Language Model

基于 PyTorch 的小型自回归语言模型学习项目。从字节级 BPE 分词开始，逐步实现 Transformer、训练、文本采样和性能测量，并通过对照实验理解模型结构与计算开销之间的关系。

当前模型采用 RoPE 和 SDPA 注意力，支持通过配置切换 LayerNorm / RMSNorm 与 GELU / SwiGLU 前馈网络。

## 已实现的功能

- **分词器**：字节级 BPE、正则预切分、特殊 token、词表与合并规则的保存和加载；使用 `Counter` 汇总重复文本片段，加快训练。
- **模型**：因果自注意力、多头注意力、向量化 QKV、SDPA、RoPE、Pre-Norm 和残差连接。
- **结构配置**：`norm_type` 选择 LayerNorm 或手写 RMSNorm，`ffn_type` 选择 GELU 前馈网络或 SwiGLU。
- **训练**：下一 token 预测、AdamW、warmup + cosine 学习率、梯度裁剪、训练集和验证集 loss 评估、早停与 checkpoint。
- **生成**：temperature、top-k、top-p、批量 EOS 停止和动态 KV cache。
- **性能测量**：warmup、CUDA 同步计时、ms/step、tokens/s、峰值显存，以及可选的 BF16 autocast。

## 文件说明

| 文件 | 用途 |
|---|---|
| `MinBPE.py` | 分词器实现：`BasicTokenizer`、`RegexTokenizer` 及保存、加载、编码、解码 |
| `BasicBPE.py` | 早期 BPE 学习实现 |
| `tiny_story_bpe.py` | 按故事边界划分语料，并在训练部分训练 tokenizer |
| `tinylanguagemodel.py` | 模型、归一化、前馈网络、采样、训练、评估和 checkpoint 保存 |
| `train.py` | 数据加载、模型配置、四组对照训练和断点续训入口 |
| `benchmark.py` | 训练和生成的性能测量；当前主入口测量四组模型的训练性能 |
| `generate.py` | 使用 checkpoint 进行生成和 KV cache 调用的实验脚本 |
| `TinyStories-valid.txt` | 本项目使用的原始 TinyStories 文本 |
| `train_text.txt` / `val_text.txt` | 本项目划分得到的训练和验证文本 |
| `tiny_story.model` / `tiny_story.vocab` | 保存的 tokenizer 规则和可读词表 |

运行时还需要本地生成的 `train_tensor.pt`、`val_tensor.pt`，以及训练产生的 checkpoint。仓库的 `.gitignore` 忽略了 `*.pt`，这些文件不会随 `git clone` 下载。

## 环境与运行

### 1. 获取项目

```bash
git clone https://github.com/nanae0430/tinylanguagemodel.git
cd tinylanguagemodel
```

以下命令都在仓库根目录执行。脚本使用相对路径读取数据和 checkpoint。

使用 Python 3.10 或以上版本，并安装 `torch` 和 `regex`。PyTorch 的安装命令按操作系统和计算平台从 [官方安装页](https://pytorch.org/get-started/locally/) 选择。

```bash
python -m pip install regex
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

如果使用 Conda，可先激活自己的环境，例如 `conda activate pytorch`。VS Code 也应选择同一个环境的 Python 解释器。

模型训练会根据 `torch.cuda.is_available()` 选择 CUDA 或 CPU。当前 benchmark 中直接调用 CUDA 同步和显存统计接口，需要可用的 CUDA GPU。

### 2. 准备数据

本项目使用 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的 `TinyStories-valid.txt`。在该文件约 90% 的位置向前寻找最近的 `<|endoftext|>`，保留完整故事边界，再划分项目自己的训练和验证文本。

Tokenizer 仅使用前面的训练文本训练。这里的验证集是本项目从该文件中划出的后约 10%，与 TinyStories 官方数据划分的含义不同。

仓库已经包含 `train_text.txt`、`val_text.txt` 和 `tiny_story.model`，换电脑后可以直接使用。如果需要重新划分数据并训练 tokenizer，运行：

```bash
python tiny_story_bpe.py
```

当前普通词表大小为 2000，另外注册 `<|endoftext|>`，ID 为 2000；包含该特殊 token 的总词表大小为 2001。模型从 `len(tokenizer.vocab)` 获取实际词表大小。

### 3. 生成 token 缓存

如果本地没有 `train_tensor.pt` 和 `val_tensor.pt`，只需要使用保存的 tokenizer 重新编码，不必重新训练 tokenizer。

下面是一次性编码片段，可临时保存为 Python 脚本，在仓库根目录运行：

```python
import torch
from MinBPE import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.load("./tiny_story.model")

for split in ("train", "val"):
    with open(f"./{split}_text.txt", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text, {"<|endoftext|>"})
    tensor = torch.tensor(ids, dtype=torch.long)
    torch.save(tensor, f"./{split}_tensor.pt")
```

### 4. 训练

```bash
python train.py
```

当前入口在 `load_checkpoint=False` 时，按下面的配置列表依次从头训练四组模型：

```python
norm_types = ["LayerNorm", "RMSNorm"]
ffn_types = ["GELU", "SwiGLU"]
```

只训练一种组合时，将列表分别缩减为所需的一项。训练超参数直接在 `train.py` 中修改，当前没有命令行参数解析。

每组使用独立的 checkpoint 文件名，例如：

- `best_checkpoint_RMSNorm_SwiGLU.pt`
- `last_checkpoint_RMSNorm_SwiGLU.pt`

Checkpoint 保存 `model_config`、模型权重、优化器状态、步数、loss 和早停状态。`norm_type`、`ffn_type` 也包含在 `model_config` 中，用于重建相同结构。

保存动作在评估时执行：当前训练函数每 1000 步或最后一步评估并保存。`last_checkpoint` 表示最近一次保存的状态。

断点续训时，将 `load_checkpoint` 改为 `True`，并检查该分支的 `path`、目标总步数和传给 `train()` 的学习率配置。旧 checkpoint 如果缺少新增的类型字段，需要按当时的模型结构补齐；归一化和前馈网络的结构应与权重匹配。

### 5. 性能测量与生成

```bash
python benchmark.py
```

当前 benchmark 从四组 `last_checkpoint_<组合名>.pt` 加载模型，因此需要先完成四组训练。计时覆盖数据取样与传输、前向、反向和优化器更新，是训练步骤的整体耗时。

Benchmark 会在内存中继续更新模型参数，但当前脚本不会把这些更新写回 checkpoint。

`generate.py` 保留了早期 RoPE checkpoint 的实验路径。使用前需要调整 checkpoint 文件名，并确保 `model_config` 包含正确的 `norm_type` 和 `ffn_type`，再运行：

```bash
python generate.py
```

生成时，`top_k=1` 使用贪心选择；`top_p` 不为 `None` 时采用 top-p 分支。动态 KV cache 将历史 K/V 与新 token 的 K/V 拼接；启用 cache 后，达到 `block_size` 上限会停止继续生成。

## RMSNorm 与 SwiGLU

### RMSNorm

对每个 token 沿最后一个特征维度计算：

$$
y_i=\gamma_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}
$$

输入与输出均为 `[B, T, D]`，分母形状为 `[B, T, 1]`，可学习参数 `gamma` 的形状为 `[D]`。当前实现使用 `eps=1e-6`，不减去均值，也不包含 bias。

手写 RMSNorm 已在本地与 `nn.RMSNorm` 对比输出、输入梯度和缩放参数梯度。FP32 随机样本测试中，记录的最大绝对误差在 $10^{-7}$ 量级；这属于已测试样本的结果，不是所有输入的误差上界。

### SwiGLU

将同一个输入分别送入两个独立投影，gate 分支经过 SiLU，再逐元素相乘并送入 down 投影：

$$
u=\operatorname{Linear}_{up}(x),\qquad
g=\operatorname{SiLU}(\operatorname{Linear}_{gate}(x))
$$

$$
y=\operatorname{Dropout}(\operatorname{Linear}_{down}(g\odot u))
$$

其中 $\operatorname{SiLU}(z)=z\sigma(z)$。Gate 的值由当前输入决定，可以减弱、放大或改变特征符号，其取值不限制在 $[0,1]$。

忽略 bias 时，GELU 前馈网络有约 $2DH$ 个参数，SwiGLU 有约 $3DH$ 个参数。原 GELU 网络使用 $H=4D$，因此 SwiGLU 取 $H=\lfloor 8D/3\rfloor$，近似对齐参数量。

| 当前 `D=128` 的前馈网络 | 中间维度 `H` | 每个 Block 的 FFN 参数量，含 bias |
|---|---:|---:|
| GELU | 512 | 131,712 |
| SwiGLU | 341 | 131,754 |

三个 Linear 当前都保留 bias。SwiGLU 已在本地验证 `[2, 4, 128]` 输入输出形状一致，输入和所有参数梯度非 `None`，且不包含 NaN 或 Inf。

## 实验记录：归一化与前馈网络对照

实验日期：2026-10-04。对应代码提交：[a23f23e](https://github.com/nanae0430/tinylanguagemodel/commit/a23f23e14c2c751c1f6430e444ebc59b6195bfc2)。以下数值来自本地运行日志。

### 配置与测量口径

| 项目 | 设置 |
|---|---|
| 数据 | 同一份 TinyStories 文本，项目内约 9:1 划分 |
| 模型 | `n_embd=128`，`num_head=8`，`num_layer=8`，`block_size=128` |
| Tokenizer 总词表 | 2001，包含 EOS |
| Dropout | 0.1 |
| 优化器 | AdamW |
| 训练 | batch size 16，每组 200 步 |
| 学习率 | 前 10 步 warmup，随后 cosine；最大 `1e-3`，最小 `1e-4` |
| 梯度裁剪 | `max_norm=1` |
| Loss 评估 | `model.eval()` 下，训练集和验证集各随机取 100 个 batch，分别求平均 |
| Benchmark | batch size 32，warmup 20 步，正式测量 200 步 |
| 精度 | FP32，autocast 未开启 |
| 计时 | CUDA 同步后开始计时，结束时再次同步 |
| 峰值显存 | warmup 后重置，统计 `max_memory_allocated() / 1024**2`，单位 MiB |
| 运行环境 | Windows、Conda、CUDA GPU；具体 GPU 型号及软件版本尚未记录 |
| 随机种子 | 脚本开头调用一次 `torch.manual_seed(39)`；四组之间没有重置随机状态 |

### 结果

| 配置 | 平均 train loss | eval loss | ms/step | tokens/s | 峰值显存 MiB |
|---|---:|---:|---:|---:|---:|
| LayerNorm + GELU | 4.507596 | 4.536057 | 85.063832 | 48,152.074785 | 509.960938 |
| LayerNorm + SwiGLU | 4.312185 | 4.381888 | 88.638324 | 46,210.260022 | 563.843750 |
| RMSNorm + GELU | 4.464720 | 4.526139 | 91.209120 | 44,907.789677 | 543.653809 |
| RMSNorm + SwiGLU | 4.324044 | 4.379157 | 95.865327 | 42,726.605195 | 597.427246 |

这里的 train loss 是最后一次评估的平均值，不是第 200 步单个训练 batch 的 loss。Benchmark 另行加载 checkpoint，并执行 warmup 和测量步骤；表中的 loss 来自此前的训练评估。

### 观察与限制

- 在 LayerNorm 下，换用 SwiGLU 后，eval loss 降低 0.154169；每步耗时增加约 4.20%，峰值显存增加约 53.88 MiB。
- 在 RMSNorm 下，换用 SwiGLU 后，eval loss 降低 0.146982；每步耗时增加约 5.10%，峰值显存增加约 53.77 MiB。
- 两组 SwiGLU 的 eval loss 仅相差 0.002731，当前结果不足以判断哪种归一化更好。
- 在本次硬件、规模和 FP32 实现下，LayerNorm + GELU 的训练速度最快、峰值显存最低；这不是所有模型和硬件上的通用排序。
- FFN 参数量接近时，SwiGLU 仍增加了分支和中间计算。具体速度和显存差异尚未用 profiler 分解；手写 RMSNorm 的结果也不能直接代表融合实现的性能。
- 每组只训练 200 步，未重复多个种子，也没有固定各组的评估 batch。单次在脚本开头设置种子不保证各组使用相同的随机初始化、训练 batch 或 dropout 掩码。
- 峰值显存指 PyTorch 分配的张量内存，不等同于 `nvidia-smi` 显示的整个进程显存。当前测量包含加载的模型、优化器和 checkpoint 张量，其结果依赖脚本的对象生命周期。

本次短训练中，SwiGLU 在两种归一化下都得到更低的验证损失，同时付出了约 4%～5% 的每步耗时和约 54 MiB 的峰值显存增量。后续将用更长训练、多随机种子和固定评估样本进一步验证。

## 后续学习计划

- [x] BPE 分词器、特殊 token 与数据划分
- [x] 自回归模型训练、checkpoint 与文本采样
- [x] 向量化 QKV、SDPA、RoPE 与动态 KV cache
- [x] RMSNorm 的实现、梯度对照与短训练
- [x] SwiGLU 的实现、模型接入与四组性能实验
- [ ] 完善可复现实验：记录硬件及依赖版本，重复多个种子，固定评估样本
- [ ] SFT 数据组织、训练目标和 loss mask
- [ ] LoRA 原理、实现及与全参数微调的对照
- [ ] 可选：静态 KV cache 与算子性能分析

## 参考资料

- [TinyStories 数据集](https://huggingface.co/datasets/roneneldan/TinyStories)
- [minbpe：字节级 BPE 的参考实现](https://github.com/karpathy/minbpe)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [PyTorch 文档](https://docs.pytorch.org/docs/stable/index.html)
