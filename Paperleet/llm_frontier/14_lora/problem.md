# LoRA 低秩更新与冻结基座前向

实现缩放低秩更新并冻结基座参数。

论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2)。定位：Section 4.1, Eq. (3): W0 + BA and alpha/r scaling；Section 4.1: random A, zero B initialization and merged inference。

## 精选核心

### 返回值要求

#### `lora_delta` — 需要实现

从 a `[r,in_features]`、b `[out_features,r]` 返回一个浮点 Tensor `[out_features,in_features]`，表示经缩放的低秩权重增量。保持共同 dtype/device 及两个 adapter 的梯度。仅返回增量矩阵，不合并进或修改基座权重。

#### `lora_linear` — 需要实现

对 x `[...,in_features]` 返回一个浮点 Tensor `[...,out_features]`，保留所有前导批/token 维度。其值包含基座线性输出和 adapter 贡献，可选 bias 仅加入一次。保持 x 的 dtype/device 及 x、a、b 的梯度；weight 和可选 bias 即使需要梯度也应冻结。不修改输入，也不返回更新后的参数。

### 公式与约定

实现线性层的两个函数；初始化与 rank-2 回归训练循环已提供。

$$
y=xW_0^\top+\frac\alpha r xA^\top B^\top
$$

- `lora_delta(a: Tensor, b: Tensor, alpha: float=1.0) -> Tensor`
- `lora_linear(x: Tensor, weight: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, bias: Tensor | None=None) -> Tensor`

## 接口与约定

实现 `lora_delta(A,B,alpha)` 与 `lora_linear(x,W,A,B,alpha,bias)`。形状为 `A[r,in]`、`B[out,r]`、`W[out,in]`、`x[...,in]`。增量为 `(alpha/r) * B @ A`。前向等于冻结基座线性映射加低秩更新。即使输入基座权重与 bias 的 `requires_grad=True`，也不能向它们传播梯度；`x`、`A`、`B` 的梯度必须保留。

训练脚手架随机初始化 A、零初始化 B。因此初始输出不变，B 有学习信号而 A 的首次梯度为零。对整个基座计算使用 `no_grad()` 会错误删除部分 x 梯度。将 delta 加到 W 即可合并，测试比较合并前后结果。本模块不包含 dropout。
