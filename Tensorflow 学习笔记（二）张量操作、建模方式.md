# Tensorflow 学习笔记（二）张量操作、建模方式

* 本笔记将详细介绍张量的概念、基本创建和操作、常用层以及三种建模方式。



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
  * **Sequential model**（顺序模型）：使用 Sequential 方法构建。
  * **Functional model**（函数模型）：如果模型有多输入或者多输出，或者模型需要共享权重、具有残差连接等非顺序结构，推荐使用函数式 API 进行创建。
  * **Subclassing model**（子类化模型）：需要自定义层之间的传输，模型复杂时使用。
* 用于模型训练的重要参数 tf.keras.Model.fit()：
  * **epochs**：训练分为几个时期，每一个epoch是对整个输入数据的一次迭代（此操作以较小的批次完成）。
  * **batch_size**：当传递 Numpy 数据时，模型将数据切成较小的批次，并在训练期间对这些批次进行迭代。该整数指定每个批次的大小。请注意，如果不能将样本总数除以批次大小，则最后一批可能会更小。
  * **validation data**：在模型训练时，监控在某些验证数据上监视其性能。传递此参数（输入和标签的元组）可以使模型在每个时期结束时以推断模式显示所传递数据的损失和度量。



### 顺序模型 Sequential Model

* 顺序模型类似搭积木的方式构建。需要用到 tf.keras.Sequential 方法，是 keras 自带的一种方法。



### Experiment 5：顺序模型构建

* 顺序模型一般有两种构建方式，第一种是使用 model.add 方法设置层：

```python
from tensorflow.keras import layers
import tensorflow as tf

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu')) # 第一层
model.add(layers.Dense(64, activation='relu')) # 第二层
model.add(layers.Dense(10)) # 第三层
```

* 第二种是直接写成一个 list 传给 Sequential 方法：

```python
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32, )), # 第一层
    layers.Dense(64, activation='relu'), # 第二层
    layers.Dense(10) # 第三层
])
```

* 训练模型：

```python
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)
"""
Train on 1000 samples
Epoch 1/10
  32/1000 [..............................] - ETA: 3:11 - loss: 12.1850 - accuracy: 0.1562
...
"""
```



### 函数式模型 Functional Model

* 函数式模型是一种创建模型的方法，该模型比 tf.keras.Sequential 更灵活。
* 函数式模型可以处理具有非线性拓扑的模型，具有共享层的模型以及具有**多个输入或输出**的模型等等。
* 深度学习模型通常是层的**有向无环图**（DAG）的主要思想。因此，函数式模型是一种**构建层图**的方法。



### Experiment 6：函数式模型构建

#### 基本构建

* 输入：(input: 32-dimensional vectors)
* 第一层：[Dense (64 units, relu activation)]
* 第二层：[Dense (64 units, relu activation)]
* 第三层：[Dense (10 units, softmax activation)]
* 输出：(output: logits of a probability distribution over 10 classes)

```python
inputs = tf.keras.Input(shape=(32, ))
x = layers.Dense(64, activation='relu')(inputs) # 第一层
x = layers.Dense(64, activation='relu')(x) # 第二层
predictions = layers.Dense(10)(x) # 第三层
```

```python
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.fit(data, labels, batch_size=32, epochs=5)
```

#### 多输入模型构建

```python
inputs1 = tf.keras.Input(shape=(32, ))  # 输入1
inputs2 = tf.keras.Input(shape=(32, ))  # 输入2
x1 = layers.Dense(64, activation='relu')(inputs1) # 第一层
x2 = layers.Dense(64, activation='relu')(inputs2) # 第一层
x = tf.concat([x1,x2], axis=-1) # 合并
x = layers.Dense(64, activation='relu')(x) # 第二层
predictions = layers.Dense(10)(x) # 第三层
```

```python
model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

import numpy as np
data1 = np.random.random((1000, 32))
data2 = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.fit((data1,data2), labels, batch_size=32, epochs=5)
```



### 子类模型 Subclassing Model

* 通过子类化 tf.keras.Model 和定义自己的**前向传播**模型来构建完全可定制的模型。
* 和 eager execution 模式相辅相成。



### Experiment 7：子类模型构建

```python
class MyModel(tf.keras.Model):
    
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 在这里定义自己需要的层
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes)
        
    def call(self, inputs):
        # 定义前向传播
        # 使用 __init__ 定义的层
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x
```

```python
model = MyModel(num_classes=10)

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

* 总的来说，定义一个导出类，继承自 tf.keras.Model，其中修改两个函数，详见代码。



* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.20