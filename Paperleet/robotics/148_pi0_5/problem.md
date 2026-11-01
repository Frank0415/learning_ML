# π0.5 混合动作头防泄漏注意力

## 1. 问题

π0.5 混合动作头防泄漏注意力：实现下列接口并通过数值、边界和实验检查。

实现论文附录E/Figure18的训练注意力连接；prefix、离散FAST标签和连续动作expert的内容由调用者给定，省略模型预训练。

## 2. 核心公式

$$
M_{qk}=\begin{cases}k<P&q<P\\k<P\ \lor\ (P\le k\le q)&P\le q<P+F\\k<P\ \lor\ k\ge P+F&q\ge P+F\end{cases},\quad M\leftarrow M\land v_q\land v_k.
$$

## 3. 公式到代码

- `hybrid_attention_mask(valid: Tensor, prefix_length: int, fast_length: int) -> Tensor`

prefix仅看prefix；FAST看prefix与当前之前FAST；expert看prefix与所有expert。FAST包含干净动作标签，若允许expert读取，会在流匹配训练中泄漏答案。query与key两侧都需屏蔽padding。

## 4. 本题约定

valid为(B,L)布尔，prefix_length>=1，fast_length>=0且至少保留一个expert token。输出(B,L,L)布尔，行是query，列是key。FAST为已右移的teacher-forcing输入，可见自身；expert双向但不可见FAST。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
