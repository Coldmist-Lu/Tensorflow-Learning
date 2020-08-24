# Tensorflow 学习笔记（四）自定义模型训练

* 本笔记将介绍如何在 Keras 模型下构建和训练网络模型。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：2 Model Training on Keras 记事本文件。



## 自动求导机制

### GradientTape

* 本节将介绍自动求解梯度的有效工具：**tf.GradientTape**
* GradientTape 是 eager 模式下计算梯度用的，而 eager 模式是 Tensorflow 2.0 的默认模式。

#### Experiment 1：求导机制简单例子

* 下面的例子求出了 y = x^2 在 x=3 时的导数：

```python
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x) # watch:确保某个Tensor被tape追踪
    y = x * x
dy_dx = g.gradient(y, x) # gradient:根据tape计算某个或某些Tensor的梯度
```



### tf.GradientTape 方法

```python
tf.GradientTape(persistent=False, watch_accessed_variables=True)
```

#### persistent 参数

* 用来指定新创建的 Gradient tape 是否是可持续性的。默认是 False，意味着只能够调用一次 gradient() 函数。

#### watch_accessed_variables 参数

* 表明这个 GradientTape 是不是会自动追踪任何能被训练（trainable）的变量。
* 默认是 True。要是为 False 的话，意味着需要手动去指定你想追踪的那些变量。



### gradient 方法

```python
gradient(target, sources)
```

#### target 参数

* 被微分的 Tensor，可以理解为 loss 值（针对深度学习训练来说）

#### sources 参数

* Tensors 或者 Variables 列表（当然可以只有一个值）

#### 返回值

* 一个列表表示各个变量的梯度值，和 source 中的变量列表一一对应，表明这个变量的梯度。





