# TD-MPC2 单纯形潜变量与精英轨迹更新

## 1. 问题

TD-MPC2 单纯形潜变量与精英轨迹更新：实现下列接口并通过数值、边界和实验检查。

提取SimNorm与MPPI已选精英的加权矩更新；候选轨迹与价值作为输入，不学习动力学/Q网络或执行真实MPC。

## 2. 核心公式

$$
z_{g,:}=\operatorname{softmax}(x_{g,:}),\quad w_i=\operatorname{softmax}(\tau Q_i),\quad\mu=\sum_iw_i a_i,\quad\sigma=\operatorname{clip}(\sqrt{\sum_iw_i(a_i-\mu)^2},\sigma_{min},\sigma_{max}).
$$

## 3. 公式到代码

- `simnorm(values: Tensor, group_dim: int) -> Tensor`
- `elite_moments(actions: Tensor, values: Tensor, temperature: float, min_std: float, max_std: float) -> tuple`

SimNorm在每个小组内归一化，不能让整个latent共享一个softmax。轨迹权重沿精英E维，对整条时域共享。官方temperature乘价值；方差围绕加权均值，最后再夹标准差。公式用精确归一化，省略源码分母的1e-9数值保护。

## 4. 本题约定

simnorm输入任意rank>=1浮点张量，末维被group_dim整除且group_dim>=2。actions为(H,E,D)，values为(E,)，temperature>=0且为乘数；0<min_std<=max_std。返回(H,D)的mean/std。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
