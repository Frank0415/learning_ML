# ACT 动作块时间集成与风格正则

## 1. 问题

实现对当前动作的重叠预测集成，以及条件VAE的标准正态KL。

实现ACT的两个张量核心；输入已经对齐到同一执行时刻，省略视觉Transformer、动作块网络和遥操作系统。

## 2. 核心公式

$$
\bar a=\frac{\sum_{i=0}^{K-1}e^{-ki}a_i}{\sum_i e^{-ki}},\quad L_{KL}=\frac1{2B}\sum_{b,d}(\mu_{bd}^2+e^{\ell_{bd}}-1-\ell_{bd}).
$$

## 3. 公式到代码

- `temporal_ensemble(actions: Tensor, valid: Tensor, decay: float) -> Tensor`
- `style_kl(mu: Tensor, logvar: Tensor) -> Tensor`

官方实现按旧到新索引使用负指数；因此正decay给较早预测更大权重。显式valid替代源码的全零哨兵，避免把合法零动作丢掉。KL中的logvar是对数方差。

## 4. 本题约定

actions为(K,D)，valid为(K,)布尔，至少一条有效；有效预测按生成时间从旧到新排列，先去padding再按0..Kvalid-1赋权，k>=0。mu/logvar为(B,Z)，KL按潜变量求和、按batch平均。保留浮点输入梯度。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
