# Muon Split：逐头矩阵正交化

来源：[GLM-5 技术报告 v2](https://arxiv.org/abs/2602.15763v2)，§2.1 “Multi-latent Attention”。报告将注意力上投影矩阵 $W^{UQ}, W^{UK}, W^{UV}$ 按头拆开，再分别做矩阵正交化，让各个头获得独立的更新尺度。

## 需要实现

### 返回值要求

#### `headwise_orthogonalize` — 需要实现

返回一个与 matrix 同形状/dtype/device 的浮点 Tensor `[num_heads*head_dim,input_dim]`。逐头部分极分解因子按原有连续行头顺序拼回，不返回列表或新增头维度的 Tensor。声明的相对阈值以下奇异方向贡献零，全零头保持为零。仅要求满秩且奇异值互异的块保留 autograd；秩边界/重复奇异值不属于梯度要求。不修改 matrix。

### 公式与约定

只实现 `headwise_orthogonalize(matrix, num_heads)`。

输入形状为 `[num_heads * head_dim, input_dim]`。每个头占用**连续的行块**：拆成 `[num_heads, head_dim, input_dim]`，逐头正交化，再按原顺序拼回。调用者会对每种上投影的梯度／动量矩阵分别应用这个函数。

对一个头的矩阵 $G_h=U_h\operatorname{diag}(s_h)V_h^\top$，返回

$$
P_h=U_h\operatorname{diag}(\mathbf 1[s_h>\tau_h])V_h^\top,
\qquad \tau_h=\max(d_h,d_{in})\,\epsilon_{dtype}\,\max(s_h).
$$

使用 reduced SVD。全零头返回全零；秩亏矩阵只保留非零奇异方向，不能凭空在零空间里产生更新。正比例放大某个满秩头，不改变其输出。宽的满行秩矩阵满足 $P_hP_h^\top=I$；高的满列秩矩阵满足 $P_h^\top P_h=I$。

输入为 CPU 上的 float32/float64 张量。保持形状和 dtype，不修改输入。非二维、非浮点、空维度、非正或不能整分行数的头数应抛出 `ValueError`。奇异值互不相同的满秩输入应保留正常 autograd 路径。通用 SVD backward 在重奇异值处可能无定义；这些位置与秩变化处的导数不属于本题范围，但前向结果仍有定义。

## 本题的精确边界

**本题用 SVD 计算精确的部分极分解因子。** 报告没有完整固定生产实现的核函数、多项式系数、迭代次数、数值秩处理或全部优化器参数。本题的 SVD 和阈值是明确声明的教学约定，用来单独检验逐头拆分思想；不声称复现某个近似 Newton–Schulz 核。动量形成、学习率缩放、权重衰减与训练循环都不需要实现。
