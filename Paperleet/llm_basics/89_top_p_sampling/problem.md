# Top-p：累计概率与核采样候选

可独立完成的大模型基础练习，无需先做完其他题。配套代码和合成实验已提供。

## 任务与标准约定

logits 为 CPU 上有限 float32/64 [...,V]，V≥1，前导维可为空；p 有限且 0<p≤1，temperature 有限且>0。先计算全词表 softmax(logits/temperature)，按概率降序排列，并列时词表编号小者优先。保留累计概率首次达到或超过 p 的最短非空前缀，必须包含跨过阈值的 token；p=1 显式保留全部。保留项重新归一化，再恢复原词表顺序；其他位置为零，shape/dtype/device 不变。不随机抽样、不修改输入、不要求梯度；缩放、概率和累计阈值比较均使用 float64，不加任意容差；仅最终结果转回输入 dtype。

```python
top_p_probs(logits: Tensor, p: float, temperature: float = 1.) -> Tensor
```

$$
K=\min\{k:\sum_{i=1}^k p_{(i)}\ge p\},\quad q_{(i)}=\frac{p_{(i)}\mathbf{1}[i\le K]}{\sum_{j=1}^Kp_{(j)}}
$$

## 示例

```python
top_p_probs(torch.zeros(4), .5)
# tensor([.5, .5, 0., 0.])
```


## 数值范围与实际抽样

中间量使用 float64，避免很小的 Python temperature 在 float32 中变成零。temperature ≥1 时先除后减最大值，避免异号大数先相减溢出；temperature <1 时先减后除，让最大项保持零。仅最终概率转回输入 dtype。



## 论文定义与精确边界

本题采用 [Holtzman 等，ICLR 2020，§3.1 公式 (2–3)](https://arxiv.org/abs/1904.09751) 的最短前缀定义：累计概率 ≥p。[作者发布代码](https://github.com/ari-holtzman/degen/blob/0acfd2d0ba8484e24e9c5241f75f34be15ef2609/gen.py#L118-L132) 对严格 >p 的掩码右移，在恰好相等时会多留一个 token。本题四项均匀分布、p=.5 时恰好保留两项。排序与累计比较使用 float64，精确并列优先较小词表索引。这是明确的教学约定。
