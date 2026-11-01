# Top-k：温度缩放与候选概率

可独立完成的大模型基础练习，无需先做完其他题。配套代码和合成实验已提供。

## 任务与标准约定

logits 为 CPU 上有限 float32/64 [...,V]，V≥1，前导维可为空；k 是 1≤k≤V 的整数，temperature 有限且>0。每行按最后维选恰好 k 个最大 logit；相同值优先词表编号较小者。返回同 shape/dtype/device 的概率：在选中位置对 logits/temperature 做 softmax，其余严格为零，各行和为一。每行独立处理。不随机抽 token，不修改输入，不要求梯度。恰好 k 个的并列规则是本题约定，有些阈值式实现会保留更多并列项。

缩放中间量使用 float64，仅最终概率转回输入 dtype。

```python
top_k_probs(logits: Tensor, k: int, temperature: float = 1.) -> Tensor
```

$$
q_i=\frac{\mathbf{1}[i\in K]\exp((z_i-\max z)/\tau)}{\sum_{j\in K}\exp((z_j-\max z)/\tau)}
$$

## 示例

```python
top_k_probs(torch.tensor([2., 2., 1.]), 1)
# tensor([1., 0., 0.])
```


## 数值范围与实际抽样

中间量使用 float64，避免很小的 Python temperature 在 float32 中变成零。temperature ≥1 时先除后减最大值，避免异号大数先相减溢出；temperature <1 时先减后除，让最大项保持零。仅最终概率转回输入 dtype。
