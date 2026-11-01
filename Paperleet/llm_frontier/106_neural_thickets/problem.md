# 106. Neural Thickets：随机专家筛选与多数投票

## 问题

实现 [RandOpt](https://arxiv.org/abs/2603.12228) 的核心流程：独立扰动基础权重、筛选表现最好的专家，再对离散答案投票。选择集分数与测试预测分开输入；本题不进行迭代 ES 更新，也不平均模型权重。

## 核心公式

论文第 4 节式 (3)–(5) 和 Algorithm 1 对应：

$$
\theta_i=\theta+\sigma_i\epsilon_i,\qquad
I=\operatorname{TopK}(s_1,\ldots,s_N),
$$

$$
\hat y_b=\operatorname{mode}\{y_{i,b}:i\in I\}.
$$

调用方提供噪声、每个候选的尺度、选择集分数和解析后的类别编号。投票对象是每个专家的最终答案，不是 token、概率或参数坐标。

## 公式到代码

- `perturb_candidates(theta, noise, scales)`：参数形状为 `S`，噪声 `[N,*S]`，尺度 `[N]`，返回 `[N,*S]`。每个候选都从同一基础参数出发。
- `select_experts(scores, k)`：返回按分数降序排列的 long 索引 `[k]`。
- `majority_vote(predictions, selected, num_classes)`：预测为 long `[N,B]`，返回每个样本的一个类别，形状 `[B]`。

例如分数 `[0.9,0.1,0.8]`、k=2 时选择 `[0,2]`。这两位专家分别回答 `[2,0]`、`[2,1]`，输出 `[2,0]`；第二个样本平票，选择较小的类别编号。

## 本题约定

- 参数、噪声和尺度为同 dtype/device 的有限 CPU float32/float64，中间运算有限。参数可为标量或非空形状 `S`，支持非连续视图。候选输出 detached，不修改任何输入。
- `N>=1`，尺度非负，`1<=k<=N`。分数并列时，原索引较小者优先，这是本题明确采用的确定性约定。
- 预测为 `[0,num_classes)` 内整数，`num_classes>=1`。选中索引非空且互异，每位专家一票；票数并列时取最小类别，同样属于本题约定。
- `B=0` 返回空 long 向量。不要求文本解码、答案字符串归一化或输入检查。测试标签不得参与筛选。
