# GSPO 序列比率与截断

实现序列重要性比率和截断策略损失。

论文：[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071v1)。定位：Section 4.1, Eqs. (6–8): sequence objective and geometric importance ratio；Section 4.2, Eq. (11): equal token gradient weights。

## 精选核心

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `sequence_ratio` · 需要实现

对 `[B, L]` 分数/mask（`B` 个回答、`L` 个 token 位置），返回浮点 Tensor `ratio`，形状 `[B]`。每项是有效 token 重要性比率的几何平均；空回答返回 `1`。保留输入浮点 dtype/device 及 `logp` 梯度，`old_logp` 固定；不对 batch 继续聚合。

#### `gspo_loss` · 需要实现

返回形状 `()` 的浮点标量 Tensor `loss`，将截断序列目标加负号并对 **全部 `B` 个回答** 平均，空回答以零参与平均。保留 `logp` 梯度，固定 `old_logp` 和 advantage；全空 mask 返回可反传的零 Tensor，不是 Python 数字。

### 公式与约定

只实现两个函数；组 advantage 计算、采样和分布式 RL 不属于本模块。

$$
s_i=\exp\left(\frac{\sum_t m_{it}(\log\pi_{it}-\log\pi^{old}_{it})}{\sum_t m_{it}}\right)
$$

- `sequence_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor`
- `gspo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float=0.2) -> Tensor`

## 接口与约定

`logp`、`old_logp`、二元 `mask` 形状为 `[B,L]`，`advantages` 为 `[B]`。先对有效 log-ratio 求和、除以有效长度，再取指数。每个回答只截断这个序列比率，与冻结 advantage 相乘取较小项，最后对 batch 求负均值。

旧策略与 advantage 不参与梯度。全 padding 回答比率为 1、损失贡献为 0，仍计入 batch 大小。必须在指数运算前屏蔽 padding。重复一个回答内部的 token 不改变结果。本题实现 GSPO，不是 GSPO-token。默认 `.2` 截断是便于观察的教学设置，不代表 Qwen 生产训练参数。
