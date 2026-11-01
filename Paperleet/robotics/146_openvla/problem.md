# OpenVLA 动作token与分位数还原

## 1. 问题

OpenVLA 动作token与分位数还原：实现下列接口并通过数值、边界和实验检查。

复现官方ActionTokenizer的离散约定及动作反归一化；不加载7B模型。边界数组可缩小以便手算，默认论文规模为256个边界、255个中心。

## 2. 核心公式

$$
d=\sum_j\mathbf1[\operatorname{clip}(a,-1,1)\ge e_j],\quad t=V-d,\quad \hat a=(e_i+e_{i+1})/2,\ i=\operatorname{clip}(V-t-1,0,N-2).
$$

## 3. 公式到代码

- `encode_actions(actions: Tensor, edges: Tensor, vocab_size: int) -> Tensor`
- `decode_actions(tokens: Tensor, edges: Tensor, vocab_size: int) -> Tensor`
- `unnormalize_actions(actions: Tensor, q01: Tensor, q99: Tensor, mask: Tensor) -> Tensor`

编码使用末端词表的倒序编号；np.digitize默认right=False使恰在边界上的值进入右侧桶。最大边界会得到最后一个特殊编号，但解码夹到最后中心。反归一化只作用于mask指定的坐标，未指定坐标原样保留。

## 4. 本题约定

actions为任意非空浮点张量，edges为严格递增一维且两端为-1/+1，vocab_size>=边界数。token为int64；解码token必须在[0,V)，超出动作区间的词表token按官方clip映射端点中心。反归一化输入(B,D)，q01/q99为(D,)且q99>q01，mask为(D,)布尔。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。
