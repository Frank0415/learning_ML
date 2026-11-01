# Sarathi-Serve Decode 优先与分块 Prefill

## 1. 问题

先为所有decode保留token预算，再为队首prefill分配不超过上限的块。

一次混合批次的预算分配和多轮回放；以固定token预算与最大块长替代论文的性能模型。

原文: [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) · Section 4.1–4.2; Algorithm 3, stall-free batching.

## 2. 核心公式

$$
B_{prefill}=B-|D|,\qquad c_i=\min(r_i,C,B_{remaining}),\qquad |D|+\sum_i c_i\le B.
$$

## 3. 公式到代码

- `hybrid_batch(decodes: list[int], prefills: list[int], budget: int, chunk_size: int) -> dict` — Decode IDs, scheduled (prefill_index,token_count) pairs, and remaining tokens.

先扣除所有decode的一个token，再依次处理prefill。若长prompt用完本轮块长但尚未完成，它保留队首位置。不要为新prompt挤掉已有decode，也不要一次执行完整长prompt。

## 4. 本题约定

decodes 是唯一非负请求ID列表；prefills 是按FIFO排列的正剩余token数。budget>=len(decodes)，chunk_size>=1。每个decode本轮运行一次。队首prefill消耗 min(剩余,chunk_size,预算)；未完成则停止接纳后续prefill，本轮每个请求最多一块。返回dict含decodes、prefills(index,count)、remaining(含零完成项)。
