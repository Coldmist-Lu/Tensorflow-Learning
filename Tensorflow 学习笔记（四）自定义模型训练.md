# Tensorflow 学习笔记（四）自定义模型训练

* 本笔记将介绍如何通过 eager 模式的自动求导机制来构建自定义模型，并完成训练任务。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：2 Model Training on Keras 记事本文件。



## 自动求导机制

### GradientTape

* 本节将介绍自动求解梯度的有效工具：**tf.GradientTape**
* GradientTape 是 eager 模式下计算梯度用的，而 eager 模式是 Tensorflow 2.0 的默认模式。



### Experiment 1：求导机制简单例子

* 下面的例子求出了 y = x^2 在 x=3 时的导数：

```python
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x) # watch:确保某个Tensor被tape追踪
    y = x * x
dy_dx = g.gradient(y, x) # gradient:根据tape计算某个或某些Tensor的梯度
# <tf.Tensor: id=11, shape=(), dtype=float32, numpy=6.0>
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

* 一个列表表示各个变量的梯度值，和 source 中的变量列表**一一对应**，表明这个变量的梯度。





## 自定义模型训练

### 网络求导常用示例

* 下面的例子在没有数据模型的情况下是不能运行的，这里为了举例说明对网络求导的用法：

```python
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

with tf.GradientTape() as tape:
    predictions = model(data)
    loss = loss_object(labels, predictions)
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

* 一般在网络中使用时，不需要显式调用 watch 函数，使用默认设置，GradientTape 会监控可训练变量。



### 自定义模型训练步骤

* 本节将介绍一下自定义模型训练的基本步骤。

#### 构建模型（神经网络的前向传播）

* 这里使用子类模型构建：

```python
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu') # 隐藏层
        self.dense_2 = tf.keras.layers.Dense(num_classes) # 输出层

    def call(self, inputs):
        # 定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        return self.dense_2(x)
```

* 定义数据：

```python
import numpy as np
# 10分类问题
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
```

#### 编译与自动求导

* 介绍一下详细步骤：定义损失函数 --> 定义优化函数 --> 定义 tape --> 模型得到预测值 --> 前向传播得到 loss --> 反向传播 --> 用优化函数将计算出来的梯度更新到变量上面去：

```python
model = MyModel(num_classes=10)
# 定义损失函数
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
# 定义优化函数
optimizer = tf.keras.optimizers.Adam()
# 定义tape
with tf.GradientTape() as tape:
    predictions = model(data) # 模型得到预测值
    loss = loss_object(labels, predictions) # 前向传播得到loss
# 反向传播
gradients = tape.gradient(loss, model.trainable_variables)
# 用优化器更新变量
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

* 查看可训练变量：

```python
model.trainable_variables
```



### Experiment 2：使用 GradientTape 自定义训练模型

#### 构建模型和参数

```python
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        # 定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        return self.dense_2(x)
```

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
```

* 这里需要注意，建立训练数据集的时候，需要用到：
  * tf.data.Dataset.from_tensor_slices 方法
  * .shuffle() 方法对数据进行打乱。
* 此外，还要注意，由于训练中需要用到损失函数以及优化器，这些优化器要事先进行选定，并保存成一个适当的对象，后面会用到。（例如下面代码中的 optimizer、loss_fn）

```python
model = MyModel(num_classes=10)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Prepare the training dataset.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
```

#### 自定义训练

* 这里简单的介绍一下自定义训练函数的编写：
  * 自定义训练需要一个 for 循环，每次循环进行一次 epoch。
  * 循环中每次按照 batch_size 大小遍历数据集。
  * 第二重循环内，进行梯度的自动计算，步骤还是按照前面所述，前向传播 --> 计算损失 --> 求梯度 --> 按照梯度更新参数。
  * 每 200 步打印一次结果。

```python
epochs = 3
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    
    # 遍历数据集的batch_size
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        # 打开GradientTape以记录正向传递期间运行的操作，这将启用自动区分。
        with tf.GradientTape() as tape:

            # 运行该模型的前向传播。 模型应用于其输入的操作将记录在GradientTape上。
            logits = model(x_batch_train, training=True)  # 这个minibatch的预测值

            # 计算这个minibatch的损失值
            loss_value = loss_fn(y_batch_train, logits)

        # 使用GradientTape自动获取可训练变量相对于损失的梯度。
        grads = tape.gradient(loss_value, model.trainable_weights)

        # 通过更新变量的值来最大程度地减少损失，从而执行梯度下降的一步。
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # 每200 batches打印一次.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))
```

```python
"""
Start of epoch 0
Training loss (for one batch) at step 0: 24.458969116210938
Seen so far: 64 samples
Start of epoch 1
Training loss (for one batch) at step 0: 19.38613510131836
Seen so far: 64 samples
Start of epoch 2
Training loss (for one batch) at step 0: 17.059972763061523
Seen so far: 64 samples
"""
```



### Experiment 3：自定义模型进阶（加入评估函数）

* 本节将 metric 的计算加入自定义模型中。下面是一些重要的添加函数和方法：
  * 循环开始时初始化 metrics；
  * metrics.update_state()：每 batch 之后更新；
  * metrics.result()：显示 metrics 的当前值；
  * metrics.reset_states()：清除 metrics 状态，通常用于每个 epoch 结尾。

#### 构建模型

```python
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)
    
    def call(self, inputs):
        #定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        return self.dense_2(x)
```

#### 编译模型

* 这里我们定义了训练集以及验证集，因此后续定义参数的时候也必须分别定义所使用的参数：

```python
import numpy as np
x_train = np.random.random((1000, 32))
y_train = np.random.random((1000, 10))
x_val = np.random.random((200, 32))
y_val = np.random.random((200, 10))
x_test = np.random.random((200, 32))
y_test = np.random.random((200, 10))
```

#### 定义参数

* 准备参数的时候这里需要增加一步准备 metrics 参数，通过调用 tf.keras.metrics 库来实现：

```python
# 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# 损失函数
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 准备metrics函数
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# 准备训练数据集
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 准备测试数据集
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)
```

#### 循环训练

* 循环训练中需要注意两个不同的环节：
  * 指标函数的更新、显示与重置；
  * 验证集中的验证、更新与结果显示。

```python
model = MyModel(num_classes=10)
epochs = 3
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # 遍历数据集的batch_size
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                
        # 对每一个batch
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) # 通过优化器更新权重

        # 更新训练集的metrics
        train_acc_metric(y_batch_train, logits) # 更新训练集的logits     
            
    # 在每个epoch结束时显示metrics。
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))
    
    # 在每个epoch结束时重置训练指标
    train_acc_metric.reset_states() # 非常重要，不能漏掉！

    # 在每个epoch结束时运行一个验证集。
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val)
        # 更新验证集metrics
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    print('Validation acc: %s' % (float(val_acc),))
    val_acc_metric.reset_states()
```

```python
"""
Start of epoch 0
Training acc over epoch: 0.10100000351667404
Validation acc: 0.0949999988079071
Start of epoch 1
Training acc over epoch: 0.09799999743700027
Validation acc: 0.0949999988079071
Start of epoch 2
Training acc over epoch: 0.09600000083446503
Validation acc: 0.09000000357627869
"""
```





* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.25







