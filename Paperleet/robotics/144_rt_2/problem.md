# RT-2 动作词表约束解码

## 1. 问题

RT-2 动作词表约束解码：实现下列接口并通过数值、边界和实验检查。

实现机器人任务与视觉语言任务共享logits时的输出约束；动作token集合由调用者提供，不运行PaLI-X、PaLM-E或语义推理。

## 2. 核心公式

$$
p_{bv}=\frac{\mathbf1[v\in A_b]e^{z_{bv}}}{\sum_{u\in A_b}e^{z_{bu}}},\quad A_b=A\ \text{(robot)},\ A_b=V\ \text{(VLM)}.
$$

## 3. 公式到代码

- `action_probabilities(logits: Tensor, action_tokens: Tensor, is_robot: Tensor) -> Tensor`

先建立逐行合法集合，再将非法logit替换为负无穷并做稳定softmax。先softmax再置零会破坏归一化。对机器人行的非法logits，输出及梯度都为零。

## 4. 本题约定

logits为(B,V)，action_tokens为(V,)布尔且至少一项true，is_robot为(B,)布尔；机器人行仅允许动作token，其他行允许全部词表。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
