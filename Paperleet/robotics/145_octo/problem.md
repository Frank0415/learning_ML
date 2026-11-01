# Octo 双重padding掩码归约

## 1. 问题

Octo 双重padding掩码归约：实现下列接口并通过数值、边界和实验检查。

提取官方动作头的masked_mean及时间/动作padding交集；计算未乘额外action_dim系数的逐坐标平方误差，不实现扩散网络。

## 2. 核心公式

$$
M_{bwhd}=T_{bw}\land A_{bwhd},\qquad L=\frac{\operatorname{mean}(M\odot(\hat a-a)^2)}{\max(\operatorname{mean}(M),10^{-5})}.
$$

## 3. 公式到代码

- `masked_action_loss(prediction: Tensor, target: Tensor, timestep_valid: Tensor, action_valid: Tensor) -> Tensor`

观察窗口中的padding与动作时域/维度padding含义不同，必须取交集。有效元素占比进入分母，因此追加padding不稀释损失；极端稀疏掩码遵循源码分母下界。

## 4. 本题约定

预测与目标为(B,W,H,D)，timestep_valid为(B,W)布尔，action_valid为(B,W,H,D)布尔。目标detach；全padding返回可反传的0；保留源码1e-5分母下界。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
