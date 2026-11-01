# DayDreamer 离散世界模型KL平衡

## 1. 问题

DayDreamer 离散世界模型KL平衡：实现下列接口并通过数值、边界和实验检查。

提取DayDreamer继承DreamerV2的KL balancing核心，固定alpha默认0.8；计算free-nats截断之前的KL，不实现RSSM或机器人在线训练。

## 2. 核心公式

$$
L=\alpha D_{KL}(\operatorname{sg}(q)\Vert p)+(1-\alpha)D_{KL}(q\Vert\operatorname{sg}(p)).
$$

## 3. 公式到代码

- `balanced_kl(posterior_logits: Tensor, prior_logits: Tensor, alpha: float=0.8) -> Tensor`

用log_softmax得到logq/logp。动力学项冻结整个q，包括概率与logq；表征项只冻结logp。alpha=1只更新prior，alpha=0只更新posterior。不能用一个未detach的KL乘权重代替。

## 4. 本题约定

posterior/prior logits为(B,K,C)，C>=2，alpha在[0,1]。K个独立分类潜变量的KL求和，然后按B平均。前向值与alpha无关，梯度分配依赖alpha。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
