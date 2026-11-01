# RDT-1B 统一动作槽与有效维度

## 1. 问题

把不同机器人的局部动作放进统一语义槽，并屏蔽不存在的执行维度。

提取RDT统一动作空间与采样输出mask；使用可配置小槽表，不复制完整128维schema或12亿参数Transformer。

## 2. 核心公式

$$
U_{bht}=\sum_j A_{bhj}\,\mathbf1[t=s_j],\quad M_t=\mathbf1[t\in\{s_j\}],\quad A^{exec}=\hat U\odot M.
$$

## 3. 公式到代码

- `pack_actions(local: Tensor, slots: list, width: int) -> tuple`
- `mask_actions(actions: Tensor, valid: Tensor) -> Tensor`

统一表示中的空槽不能挤压掉：同一槽始终对应相同物理量。掩码既区分缺失维度，也在执行前清除去噪器的无效输出；源码训练MSE不等于有效维度mask损失，因此本题不替换其训练目标。

## 4. 本题约定

local为(B,H,D)，slots为长度D的Python整数list，每个槽唯一且位于[0,width)；返回(B,H,width)浮点动作及(width,)布尔mask。mask_actions接收同形动作及(width,)mask。未使用槽必须为0，动作梯度保留。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
