# 学生前缀上的 On-policy Distillation

从学生分布采样，在相同前缀上蒸馏教师全词表反馈。

论文：[On-policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649v3)。定位：Section 3, Algorithm 1: GKD and student data fraction lambda；Section 3, Eqs. (2–4): token divergences on student-generated prefixes；Appendix A.7: forward/reverse KL interpretation (separate conceptual example)。

## 精选核心

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `sample_student` · 需要实现

返回 `(tokens, prefixes)`，两个 `torch.long` Tensor 的元组，形状均为 `[batch_size, length]`，设备与 `transition_logits` 一致，无梯度。`tokens` 是 `[0, V-1]` 范围内的采样 ID；`prefixes[:, 0]` 为 BOS ID `V`，后续满足 `prefixes[:, 1:] == tokens[:, :-1]`。采样使用传入的 generator；返回的是 ID，不是概率或 log-prob。

#### `distillation_loss` · 需要实现

对 `[B, L, V]` logits 返回一个浮点标量 Tensor `loss`，形状 `()`。先沿词表 `V` 求 KL 和，再对 `[B, L]` mask 选中的有效 token 平均。`reverse` 为 KL(student || teacher)，`forward` 为 KL(teacher || student)。保留 student 梯度，固定 teacher logits；空 mask 返回连接计算图的零，保留输入浮点 dtype/device。

### 公式与约定

选取 GKD 的 lambda=1 与明确方向的 KL；提供三 token Markov 语言模型代替大型语言模型。

$$
L=\mathbb E_{y\sim p_S}[\frac1{|y|}\sum_t D_{KL}(p_S(\cdot|y_{<t})\Vert p_T(\cdot|y_{<t}))]
$$

- `sample_student(transition_logits: Tensor, batch_size: int, length: int, generator: torch.Generator) -> tuple[Tensor, Tensor]`
- `distillation_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, direction: str='reverse') -> Tensor`

## 接口与约定

实现 `sample_student` 与 `distillation_loss`。采样表为 `[V+1,V]`；第 `V` 行是 BOS，第 `k` 行是 token `k` 之后的条件分布。返回 `[B,L]` 整数 `tokens` 和 `prefixes`。首个前缀为 BOS，其后必须是上一时刻学生采到的 token。使用传入 generator，温度 1 的分类采样，不穿过离散采样反向传播。此 toy 定长、无 EOS。

损失输入为 `[B,L,V]` logits：学生与教师都在同一批学生前缀上计算，再提供 `[B,L]` 二元 mask。`reverse` 表示 KL(学生 || 教师)，`forward` 表示 KL(教师 || 学生)。对完整词表求和，仅平均有效 token，冻结教师，使用 log-softmax。全 padding 返回可微零；padding logits 可能非有限，须先清零。有效位置 logits 有限。
