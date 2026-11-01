# Megatron 列并行与行并行 MLP

## 1. 问题

对齐分片的隐藏通道，局部计算 GeLU 后归约第二层输出。

单进程模拟两层 tensor-parallel MLP，包括偏置与反向梯度；采用 tanh GeLU 近似，不实现分布式通信。

原文: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) · Section 3, Figure 3: column-partitioned first GEMM and row-partitioned second GEMM.

## 2. 核心公式

$$
Y=\sum_{r=0}^{P-1}\operatorname{GeLU}(XA_r+a_r)B_r+b.
$$

## 3. 公式到代码

- `parallel_mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, parts: int) -> Tensor` — (N,O) output with autograd through every floating input.

每个分片完整持有输入特征，仅拥有隐藏通道子集。GeLU 位于局部两次矩阵乘之间；第二层部分输出相加，不取平均。

## 4. 本题约定

x:(N,D)，w1:(D,H)，b1:(H,)，w2:(H,O)，b2:(O,)；H 被 parts 整除。第一层按列、第二层按相同隐藏通道行分片。b2 在归约后仅加入一次。所有浮点输入保留梯度、dtype 与原始值。
