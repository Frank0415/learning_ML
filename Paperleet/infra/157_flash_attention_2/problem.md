# FlashAttention-2 延后归一化与行块

## 1. 问题

按查询块流式处理 K/V，维护未归一化输出并返回 logsumexp。

单头方形注意力的 CPU 规格实现，含因果 mask、尾块、数值稳定性与自动微分；不计 GPU warp 时间。

原文: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) · Section 3.1.1, Algorithm 1; Section 3.3, query-row work partitioning.

## 2. 核心公式

$$
m\leftarrow\max(m,\max_j S_j),\quad \ell\leftarrow e^{m_{old}-m}\ell+\sum_j e^{S_j-m},\quad U\leftarrow e^{m_{old}-m}U+e^{S-m}V,\quad O=U/\ell,\quad L=m+\log\ell.
$$

## 3. 公式到代码

- `flash2_forward(q: Tensor, k: Tensor, v: Tensor, query_block: int, key_block: int, causal: bool = False) -> tuple[Tensor, Tensor]` — Attention output and per-row logsumexp.

外层遍历 Q 的行块，使每块独立拥有输出。内层对 K/V 块更新稳定最大值、分母和未归一化分子。最大值变化时同时重缩放旧分母与分子，结束后一次归一化。

## 4. 本题约定

q,k:(N,D)，v:(N,V)，返回 O:(N,V) 和 L:(N,)。分数包含 1/sqrt(D)。query_block 与 key_block 为正整数；causal=True 时只允许 key_index<=query_index。每行至少能见自身；最后才对 U 除以 l。所有输出保持梯度。
