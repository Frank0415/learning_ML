# DAgger 专家混合与访问状态聚合

## 1. 问题

实现逐状态专家混合，以及保留所有历史轮次的数据聚合。

提取 DAgger Algorithm 3.1 的数据闭环；标量线性环境中重新拟合策略，不实现完整交互式专家或分类器。

## 2. 核心公式

$$
\pi_i=\beta_i\pi^*+(1-\beta_i)\hat\pi_i,\qquad D\leftarrow D\cup\{(s,\pi^*(s))\}.
$$

## 3. 公式到代码

- `mixture_actions(expert: Tensor, learner: Tensor, draws: Tensor, beta: float) -> Tensor`
- `aggregate_dataset(old_states: Tensor, old_labels: Tensor, states: Tensor, labels: Tensor) -> tuple`

对同一状态只选择一个策略的动作，不能将两者数值平均。聚合沿样本轴追加，旧标签必须保留。专家为学习者访问的状态重新标注，标签不是实际执行的混合动作。

## 4. 本题约定

动作 expert/learner 为 (B,A)，draws 为 (B,) 且位于 [0,1)；draw<beta 时选专家。旧/新状态分别 (N,S)/(M,S)，标签 (N,A)/(M,A)；N可为0，M必须大于0。聚合结果脱离计算图；混合动作保留被选分支的梯度。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
