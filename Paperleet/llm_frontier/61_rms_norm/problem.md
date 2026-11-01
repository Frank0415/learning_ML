# RMSNorm 与部分特征归一化

## 1. 问题

### 返回值要求

#### `rms_norm` — 需要实现

返回一个浮点 Tensor，完整形状为 x 的 `[...,D]`；D 为特征宽度，保留全部前导批/token 轴。返回经缩放的归一化特征，不返回 RMS 统计量。保持 x 的 dtype/device 和 x、weight 的梯度，不修改输入。

#### `partial_rms_norm` — 需要实现

返回一个浮点 Tensor `[...,D]`；尽管只有前 k 个特征用于计算归一化尺度，仍返回全部 D 个特征并保留所有前导维度，D 为特征宽度。保持 dtype/device 及 x、weight 的梯度，包括所有输出对前缀统计量的依赖。不要 detach 统计量或修改输入。

### 公式与约定

实现 `rms_norm` 和 `partial_rms_norm`，理解归一化时保留均值会改变什么，以及只用部分特征估计尺度会带来什么影响。每个 token 独立处理，不需要构建 Transformer。

依据是 [RMSNorm 原论文](https://arxiv.org/abs/1910.07467) 的式 (4) 和第 5 节；作者为 Biao Zhang、Rico Sennrich，发表于 NeurIPS 2019。

## 2. 核心公式

对一个有 $D$ 个特征的向量，本题采用：

$$
s_D(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\varepsilon},\qquad
y_i=\frac{x_i}{s_D(x)}w_i.
$$

部分特征版本只替换统计量，仍输出全部特征：

$$
s_k(x)=\sqrt{\frac1k\sum_{j=1}^k x_j^2+\varepsilon},\qquad
y_i=\frac{x_i}{s_k(x)}w_i,\quad i=1,\ldots,D.
$$

没有减均值，也没有可学习 bias。`weight` 在归一化之后相乘，不能先乘到统计量中。

## 3. 公式到代码

沿最后一维计算平方均值，保留该维以便广播。pRMSNorm 的分母来自 `x[..., :k]`，分子仍是完整的 `x`。不要 detach 分母：即使只改变尾部输出的损失，前缀特征也会通过分母收到梯度。

可用手算检查：`x=[-3,4,12]` 的全量平方均值是 $169/3$，前两个特征的平方均值是 $25/2$。`k=D` 应恢复全量版本。

## 4. 本题约定

| 项目 | 约定 |
| --- | --- |
| 输入 | `x [..., D]`、`weight [D]`，$D\ge1$，同一浮点 dtype/device |
| 精度 | 有限 float32/float64；保留精度、形状、设备和梯度，不修改输入 |
| epsilon | 正数，放在平方根内：`sqrt(mean_square + eps)` |
| 部分统计 | `k` 是显式整数，$1\le k\le D$；非法值和 bool 抛出 `ValueError` |
| 实现范围 | 用 tensor 运算补全两个函数，不直接调用归一化库 API |

论文的主公式省略了数值稳定项。作者 2019 年的 PyTorch 代码把 epsilon 加在 RMS 外，本题选择平方根内的约定；两种约定在小输入时不同。本题直接传入 `k`，不要求复现论文与代码中百分比取整的差别。
