# RT-1 TokenLearner 空间汇聚

## 1. 问题

RT-1 TokenLearner 空间汇聚：实现下列接口并通过数值、边界和实验检查。

提取视觉token压缩器；MLP注意力logits作为输入，不训练EfficientNet、Transformer或真实机器人。

## 2. 核心公式

$$
Z_{bkc}=\sum_l\operatorname{softmax}_l(S_{blk})X_{blc}.
$$

## 3. 公式到代码

- `token_learner_pool(features: Tensor, logits: Tensor) -> Tensor`

将logits转成(B,K,L)，沿最后维归一化后乘原始features。每个学习token是一组空间凸组合，不能沿K归一化，也不应对被汇聚features再次做LayerNorm。

## 4. 本题约定

features为(B,L,C)，logits为(B,L,K)；只沿空间L做softmax。输出(B,K,C)，两输入均保留梯度。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
