# 逐位置前馈网络

注意力在不同位置之间交换信息；Transformer 随后对每个 token 独立应用同一个非线性网络。本题实现 [Attention Is All You Need §3.3，式 (2)](https://arxiv.org/html/1706.03762v7#S3.SS3)。

## 你要实现的核心

### 返回值要求

#### `positionwise_ffn` — 需要实现

返回一个浮点 Tensor `[...,D]`，完整形状与 x 相同；D 为模型特征宽度，前导批/token 轴不变。返回第二个线性层的输出，不返回宽度为 H 的隐藏激活。保持 dtype/device 及 x、w1、b1、w2、b2 的梯度，各 token 独立且不修改输入。

### 公式与约定

实现 `positionwise_ffn(x, w1, b1, w2, b2)`：

$$
\mathrm{FFN}(x)=\operatorname{ReLU}(xW_1^\top+b_1)W_2^\top+b_2.
$$

本题采用 PyTorch 的 `[out_features, in_features]` 权重布局，因此相对论文的行向量写法使用转置。两层仿射变换都有偏置，中间使用原始 ReLU，输出端没有额外激活。

| 参数/输出 | 约定 |
|---|---|
| `x` | `[..., D]`，允许任意前导 batch/sequence 维，也允许单个 `[D]` 向量 |
| `w1`、`b1` | `[H,D]`、`[H]` |
| `w2`、`b2` | `[D,H]`、`[D]` |
| 输出 | 与 `x` 的形状、dtype、device 相同 |
| 范围 | 形状兼容的有限浮点张量；不修改输入，并保留五个参数的梯度 |
| 位置独立 | 各位置共享参数，不沿 token 维求和或平均 |

原始 base 模型使用 `D=512`、`H=2048`，本题使用非方形的小维度。残差连接、LayerNorm 和 dropout 属于独立组件；本函数只做两次仿射变换和 ReLU。

例如隐藏层激活前的值为 `[-1,2,-3]`，ReLU 后得到 `[0,2,0]`；第二层再把它投影回模型维度并加上 `b2`。
