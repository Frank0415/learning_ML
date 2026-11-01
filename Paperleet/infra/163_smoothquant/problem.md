# SmoothQuant 激活与权重等价平滑

## 1. 问题

计算逐输入通道平滑因子，并验证相反方向缩放保持线性层输出与梯度。

校准最大值、离线scale与可微等价变换；实验用对称INT8模拟量化，未实现整数GEMM或LayerNorm融合。

原文: [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) · Section 4, Equations (3) and (4); official smooth.py offline scale construction.

## 2. 核心公式

$$
s_j=\max\left(\frac{(\max_i|X_{ij}|)^\alpha}{\max(\max_o|W_{jo}|,\epsilon)^{1-\alpha}},\epsilon\right),\quad \widehat X=X/s,\quad \widehat W=sW,\quad \widehat X\widehat W=XW.
$$

## 3. 公式到代码

- `smooth_scales(x: Tensor, weight: Tensor, alpha: float = 0.5, eps: float = 1e-5) -> Tensor` — Detached positive per-input-channel calibration scales.
- `smooth_transform(x: Tensor, weight: Tensor, scales: Tensor) -> tuple[Tensor, Tensor]` — Differentiable reciprocally rescaled activations and weights.

在样本轴提取激活最大值，在输出轴提取权重最大值，二者得到同一个输入通道向量。激活除以scale，权重乘以scale。校准统计无需梯度，等价矩阵乘法仍保留训练梯度。

## 4. 本题约定

x:(N,D)，weight:(D,O)，alpha在[0,1]；eps>0。权重通道绝对最大值先clamp到eps，最终scale再clamp到eps；零激活通道按该公式处理。scale脱离校准图，smooth_transform仍对x、weight、scale可微。输入不修改。
