# DeepMimic 四分量动作模仿奖励

## 1. 问题

DeepMimic 四分量动作模仿奖励：实现下列接口并通过数值、边界和实验检查。

实现姿态、角速度、末端与质心四项模仿奖励；姿态输入为预先计算的四元数测地角误差，不实现PPO、姿态解析或物理仿真。

## 2. 核心公式

$$
r^I=.65e^{-2\sum_j\theta_j^2}+.10e^{-.1\sum_j\|\Delta\dot q_j\|^2}+.15e^{-40\sum_e\|\Delta p_e\|^2}+.10e^{-10\|\Delta c\|^2}.
$$

## 3. 公式到代码

- `imitation_reward(pose_angles: Tensor, velocity_errors: Tensor, end_errors: Tensor, com_errors: Tensor) -> Tensor`

每个分量先对其所有关节/末端坐标平方求和，再应用论文特定指数系数。四项权重和为1，因此完美匹配奖励为1；相同末端偏差比角速度偏差受到更强惩罚。

## 4. 本题约定

pose_angles为(B,J)且在[0,pi]，velocity_errors为(B,J,3)，end_errors为(B,E,3)，com_errors为(B,3)，全部同dtype/device。返回(B,)奖励，不跨batch平均。所有误差保留梯度。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
