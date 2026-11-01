# SiTU-GLU 平滑有界激活

实现 Kimi K3 FFN 中的激活核心。[报告 v2 §2.3.2、式 (12)](https://arxiv.org/html/2607.24653v2#S2.SS3.SSS2) 对两条分支分别作平滑限制：

$$
c_\beta(z)=\beta\tanh(z/\beta),\qquad
\operatorname{SiTU}(g,u)=c_{\beta_1}(g)\,\sigma(g)\,c_{\beta_2}(u).
$$

默认 `beta1=4`、`beta2=25`。注意 sigmoid 使用原始 `gate`，不是 softcap 后的数值。

## 实现接口

### 返回值要求

#### `softcap` — 需要实现

返回一个与 x 的形状、dtype、device 完全相同的浮点 Tensor，也支持标量和空 Tensor。各值被平滑限制在 [-beta,beta] 范围内，数值计算可能达到端点。保留 x 的梯度，不修改 x，也不附带返回阈值参数。

#### `situ_glu` — 需要实现

返回一个浮点 Tensor，与 gate、up 的共同形状/dtype/device 相同，保留任意前导维度及空轴。它是逐元素门控激活，绝对值不超过 beta1*beta2，不归约特征维度。保留两分支的梯度，不修改输入。

### 公式与约定

| 函数 | 输入 | 输出 |
|---|---|---|
| `softcap(x, beta)` | 任意形状浮点张量与正的有限标量 | 同形状的平滑限制结果 |
| `situ_glu(gate, up, beta1=4, beta2=25)` | 两个形状相同的已投影张量 | 逐元素激活结果 |

保留 dtype、device 及两条分支的梯度，不修改输入；支持非连续张量和空的前导维度。测试使用 CPU float32/float64。参数有效性检查已放在 starter 中。无需实现输入投影、输出投影或 MoE 路由。

在原点附近，`softcap(x, beta)` 接近 `x`；两条分支同时接近零时，结果接近 SwiGLU。有限的 cap 限制每个输出坐标的绝对值，默认上界为 `4 × 25 = 100`。数学性质见报告 Appendix B。硬截断会改变梯度，因此不能代替 tanh。
