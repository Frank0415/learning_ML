# π0 反向时间流与动作积分

## 1. 问题

π0 反向时间流与动作积分：实现下列接口并通过数值、边界和实验检查。

提取官方openpi的线性流路径和Euler采样方向；显式给定噪声与时间，用线性速度回归替代VLA动作专家。

## 2. 核心公式

$$
x_t=t\epsilon+(1-t)a,\quad u=\epsilon-a,\quad x_{t+\Delta t}=x_t+\Delta t\,v_t,\quad \Delta t\le0.
$$

## 3. 公式到代码

- `flow_interpolate(actions: Tensor, noise: Tensor, time: Tensor) -> tuple`
- `euler_step(state: Tensor, velocity: Tensor, dt: float) -> Tensor`

官方代码的时间方向是动作到噪声，速度标签为noise-actions。采样时负步长把噪声拉回动作；不能同时翻转标签和遗漏负步长。detach仅针对监督标签，不针对混合状态。

## 4. 本题约定

action/noise为(B,H,D)，time为(B,)且在[0,1]。返回混合状态和detach的速度标签；状态保留梯度。Euler输入同形，dt有限且<=0。t=1为噪声，采样走向t=0。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
