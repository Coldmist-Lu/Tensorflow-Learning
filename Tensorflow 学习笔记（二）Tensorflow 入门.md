# Tensorflow 学习笔记（二）张量操作

## 张量

### 张量 Tensor

#### Tensor

* Tensor 是 Tensorflow 中**最重要的概念**：
  * Tensorflow 使用一种叫 Tensor 的数据结构来定义所有数据，可将 Tensor 看成是 n 维的 array 或 list。
  * 在 Tensorflow 的各部分图形间**流动传递**的只能是 Tensor。
  * 编写 Tensorflow 程序时，操纵和传递的主要对象是 tf.Tensor，它有两个部分组成：
    * 数据类型（float32、int32、string等）；
    * 形状 shape（表示张量的维度）。

#### rank

* rank 的概念：rank 表示 Tensor 的维度，具体定义如下：
  * rank = 0 表示标量，其 shape 是 []；
  * rank = 1 表示向量，其 shape 是 [D0]；
  * rank = 2 表示二维张量，其 shape 是 [D0, D1]；
  * 以此类推。

#### numpy 与 Tensor 的比较

* 下面的语句比较了 numpy 数组与 Tensor 的区别：

```python
import tensorflow as tf
import numpy as np

# 创建
a_tf = tf.zeros((2, 2)); b_tf = tf.ones((2, 2))
a_np = np.zeros((2, 2)); b_np = np.ones((2, 2))

# 求和
tf.reduce_sum(b_tf, axis=1)
np.sum(b_np, axis=1)

# 形状
a_tf.get_shape()
a_np.shape

# 改变形状
a_tf.reshape(a_tf, (1, 4))
a_np.reshape(a_np, (1, 4))

# 计算
b_tf*5+1
b_np*5+1

# 矩阵乘法
tf.matmul(a_tf, b_tf)
np.dot(a_np, b_np)

# 索引
a_tf[0, 0]; a_tf[:, 0]; a_tf[0, :]
a_np[0, 0]; a_np[:, 0]; a_np[0, :]
```

#### 推荐网站

* 推荐基础知识学习网站：
  * https://lyhue1991.github.io/eat_tensorflow2_in_30_days/
  * https://zh.d2l.ai/
  * https://tensorflow.google.cn/versions/r2.0/api_docs?hl=en # 原课程网址打不开
  * https://tf.wiki/zh_hans/



### Experiment 1：张量实验

#### 定义标量

```python
# rank=0
ch = tf.Variable("abc", tf.string)
tf.print(tf.rank(ch)) # 0
tf.print(tf.shape(ch)) # []
```

* 需要注意，这里使用的输出函数是 **tf.print** 而非 print，这是因为 tf.rank 和 tf.shape 的返回值都是 Tensor，如果直接运用 print 语句输出的结果将不那么直观：

```shell
# 换用print输出
ch = tf.Variable("abc", tf.string)
print(tf.rank(ch)) # tf.Tensor(0, shape=(), dtype=int32)
print(tf.shape(ch)) # tf.Tensor([], shape=(0,), dtype=int32)
```

#### 定义向量

```shell
# rank=1
chls = tf.Variable(["Hello"], tf.string)
tf.print(tf.rank(chls)) # 1
tf.print(tf.shape(chls)) # [1]
```

#### 定义二维张量

```python
# rank=2
mat = tf.Variable([[7], [11]], tf.int16)
tf.print(tf.rank(mat)) # 2
tf.print(tf.shape(mat)) # [2 1]
```

#### 其他创建方法

* **tf.constant**、**tf.zeros**、**tf.ones**、**tf.reshape** 的用法：

```python
tf.constant([1, 2, 3], dtype=tf.int16)
# <tf.Tensor: id=16, shape=(3,), dtype=int16, numpy=array([1, 2, 3], dtype=int16)>
tf.zeros((2, 2), dtype=tf.int16)
"""
<tf.Tensor: id=21, shape=(2, 2), dtype=int16, numpy=
array([[0, 0],
       [0, 0]], dtype=int16)>
"""
rank_three_tensor=tf.ones([3,4,5])
matrix=tf.reshape(rank_three_tensor,[6,10])
# ! yet_another=tf.reshape(rank_three_tensor,[8, 10]) 错误的值数量，应该为60，但这里是80
```



### 张量操作库

* 本节介绍一些用于张量基本操作的库：
* tf.strings （对 string 类型数据进行分割和操作，常用于推荐算法场景、NLP 场景）
* tf.debugging（用于 debug 和报错的库）
* tf.dtypes（创造某个数据类型对象，或进行数据类型转换）
* tf.math（数学计算库）
* tf.random（随机化函数）
* tf.feature_column（结构化数据的操作库）



### Experiment 2：张量操作实验

#### tf.strings

* 下面的三行语句分别展示了按字符切割、按单词切割、以及对字符串进行哈希的函数：

```python
tf.strings.bytes_split('hello')
# <tf.Tensor: id=111, shape=(5,), dtype=string, numpy=array([b'h', b'e', b'l', b'l', b'o'], dtype=object)>
tf.strings.split('Hello World')
# <tf.Tensor: id=176, shape=(2,), dtype=string, numpy=array([b'Hello', b'World'], dtype=object)>
tf.strings.to_hash_bucket(['hello', 'world'], num_buckets=10)
# <tf.Tensor: id=115, shape=(2,), dtype=int64, numpy=array([8, 1], dtype=int64)>
```

#### tf.debugging

* 用 tf.debugging 库来判断输入是否为某个形状：

```python
a = tf.random.uniform((10, 10))
tf.debugging.assert_equal(x=a.shape, y=(10, 10)) # 符合输入，无返回值
tf.debugging.assert_equal(x=a.shape, y=(20, 10)) # 报InvalidArgumentError错误
```

#### tf.dtypes

* 下面展示了数据类型的转变：

```python
x = tf.constant([1.8, 2.2], dtype=tf.float32)
x1 = tf.dtypes.cast(x, tf.int32)
tf.print(x1) # [1 2]
```

#### tf.math

* tf.math 库提供了一系列张量的数学运算：

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

tf.print(tf.math.add(a, b))
tf.print(tf.math.subtract(a, b))
tf.print(tf.math.multiply(a, b))
tf.print(tf.math.divide(a, b))
"""
[[6 8]
 [10 12]]
[[-4 -4]
 [-4 -4]]
[[5 12]
 [21 32]]
[[0.2 0.33333333333333331]
 [0.42857142857142855 0.5]]
"""
```

#### tf.random

* tf.random 库提供了很多用于创建张量的随机化函数：

```python
a = tf.random.uniform(shape=(10, 5), minval=0, maxval=10)
a
"""
<tf.Tensor: id=291, shape=(10, 5), dtype=float32, numpy=
array([[4.9565578 , 5.4019976 , 8.346172  , 0.3549707 , 7.6280317 ],
       [2.1049523 , 2.3415685 , 7.328106  , 7.0217743 , 1.1166883 ],
       [7.9804363 , 8.421116  , 1.8678772 , 1.9975865 , 2.9893696 ],
       [3.941077  , 5.7555866 , 6.08972   , 9.533499  , 0.649662  ],
       [3.6055958 , 6.6691437 , 0.05470276, 6.7765894 , 5.219883  ],
       [3.7550533 , 9.011452  , 4.483824  , 8.762631  , 9.914373  ],
       [3.4840047 , 0.48357725, 2.549324  , 0.37060022, 0.48536062],
       [8.919459  , 8.066789  , 6.3364697 , 9.166147  , 5.7335463 ],
       [8.517086  , 9.22727   , 0.4796493 , 6.8215466 , 9.808736  ],
       [4.399667  , 7.071593  , 1.4824975 , 0.82803965, 5.8337307 ]],
      dtype=float32)>
"""
```





## 常用层

* Tensorflow 中的神经网络层主要由下面两个库实现：
  * tf.nn：底层的函数库，其他各种库是基于这个底层库来扩展的，可以看成是"轮子"。
  * tf.keras.layers：基于 tf.nn 的高度封装，可以看成是"汽车"。
* 大多数情况下，可以使用 tf.keras.layers 构建的一些层来建模。
  * 它提供的层有：Dense、Conv2D、LSTM、BatchNormalization、Dropout 等。



### Experiment 3：文本分类实例

* 下例的网络创建了一个 LSTM 层和一个全连接层（Dense），最后以 softmax 为输出。

```python
a = tf.random.uniform(shape=(10, 100, 50), minval=-0.5, maxval=0.5)
x = tf.keras.layers.LSTM(100)(a)
x = tf.keras.layers.Dense(10)(x)
x = tf.nn.softmax(x)
```



### Experiment 4：增加层的参数配置

* 增加激活函数：

```python
tf.keras.layers.Dense(64, activation='relu')
tf.keras.layers.Dense(64, activation=tf.nn.relu) # 从nn库中调取激活函数
```

* 将 L1 正则化系数为 0.01 的线性层应用于内核矩阵：

```python
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
```

* 将 L2 正则化系数为 0.01 的线性层应用于偏差向量：

```python
tf.keras.layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
```

* 内核初始化为随机正交矩阵的线性层：

```python
tf.keras.layers.Dense(64, kernel_initializer='orthogonal')
```

* 偏差向量初始化为 2.0 的线性层：

```python
tf.keras.layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))
```





## 建模方式

* 本节将介绍三种建模方式：
  * Sequential model（顺序模型）
  * Functional model（函数模型）
  * Subclassing model（子类化模型）

### 顺序模型

* 顺序模型类似搭积木的方式构建。



