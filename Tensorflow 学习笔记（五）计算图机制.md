# Tensorflow 学习笔记（五）计算图机制

* 本笔记将介绍 Tensorflow 中的计算图，并且着重介绍 TF2.0 使用的 AutoGraph 机制。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：4 AutoGraph 记事本文件。



## 计算图

计算图有三种，分别是：静态计算图、动态计算图、AutoGraph。



### 静态计算图与动态计算图

#### 静态计算图

* 静态计算图在 tf1.x 版本中使用，程序在编译执行时，先生成神经网络的结构，在执行相应操作。
* 静态计算从理论上讲允许编译器进行更大程度的优化，但也意味着程序与编译器实际执行之间存在更多的代沟。
* 因此，代码中的错误将更加难以发现（比如，如果计算图结构出现问题，可能只有在代码执行到相应操作时才能发现。

#### 动态计算图

* 动态计算意味着程序将按照我们编写的顺序进行执行。这种机制将使得调试更加容易，将大脑中的想法转化成实际代码也更加容易。



### AutoGraph

* Tensorflow 2.0 主要使用的是动态计算图和AutoGraph。
* AutoGraph 机制可以将**动态图**转换成**静态计算图**，兼收**执行效率和编码效率**之利。
* AutoGraph 在 Tensorflow2.0 中主要通过装饰器 @tf.function 实现。



## AutoGraph 使用规范

本节将先给出 AutoGraph 的三条规范，然后逐一用实例讲解。



### 重要的三条规范

* 被 @tf.function 修饰的函数应尽量**使用 Tensorflow 中的函数**而不是 Python 中的其他函数。
* 避免在 @tf.function 修饰的函数内部**定义 tf.Variable**。
* 被 @tf.function 修饰的函数不可修改该函数**外部的 Python 列表或字典**等结构类型变量。



### Experiment 1：AutoGraph 三条规范的使用

#### 规范一：尽量使用 Tensorflow 中的函数

* 下面的例子中调用了随机初始化函数，比较一下两种方法的区别：

```python
@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)
    
@tf.function
def tf_random():
    a = tf.random.normal((3, 3))
    tf.print(a)
```

* 上面的例子中看似使用 numpy 库进行初始化没有问题，但是当重复调用时会发现每次返回的结果都是相同的，没有真正做到随机初始化：

```python
np_random()
np_random()
"""
array([[-0.05375999, -0.02984461, -0.27431426],
       [ 0.33941643, -0.65485957,  0.01208409],
       [-0.78438792,  0.55207263,  0.2093346 ]])
array([[-0.05375999, -0.02984461, -0.27431426],
       [ 0.33941643, -0.65485957,  0.01208409],
       [-0.78438792,  0.55207263,  0.2093346 ]])
"""
tf_random()
tf_random()
"""
[[0.920899928 3.45999646 -0.568055272]
 [-1.25254834 -0.174138784 1.50039983]
 [1.03998828 -1.12362742 0.126267701]]
[[-0.0553494431 -0.870174468 0.280180365]
 [-0.620883048 0.216187686 -1.38550234]
 [-0.554089487 1.73714256 0.930539]]
"""
```

#### 规范二：避免在函数内部定义 tf.Variable

* 首先展示一个正确的例子：

```python
x = tf.Variable(1.0, dtype=tf.float32)

@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return x

outer_var() # 2
outer_var() # 3
```

* 如果我们将 Variable 定义在 AutoGraph 内部就会报错：

```python
@tf.function
def inner_var():
    x = tf.Variable(1.0, dtype=tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return x

# ! inner_var() 报错
```

#### 规范三：不可修改函数外部的Python列表或字典等结构

* 下面给出的是一个对列表进行 append 操作的例子。首先不加 @tf.function：

```python
tensor_list = []

#@tf.function 这里我们不加装饰器
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
# [<tf.Tensor: id=70, shape=(), dtype=float32, numpy=5.0>, <tf.Tensor: id=71, shape=(), dtype=float32, numpy=6.0>]
```

* 下面是加上 @tf.function 的情况，可以看到 tensor_list 没有改变：

```python
tensor_list = []

@tf.function
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
# [<tf.Tensor 'x:0' shape=() dtype=float32>]
```





## AutoGraph 机制原理

* AutoGraph 执行的主要机制包括**创建计算图**和**执行计算图**两个部分。先创建，后执行。

### Experiment 2：AutoGraph 机制

* 用下面的例子来介绍：

```python
@tf.function(autograph=True)
def myadd(a, b):
    for i in tf.range(3):
        tf.print(i)
    c = a + b
    print("tracing")
    return c

myadd(tf.constant('hello'), tf.constant('world'))
"""
tracing
0
1
2
<tf.Tensor: id=49, shape=(), dtype=string, numpy=b'helloworld'>
"""
```

* 由于 AutoGraph 先创建计算图，后执行计算图，因此会先打印与 tf 无关的 print 语句，所以 "tracing" 被打印出来；然后经过计算图计算以后通过 TensorFlow 的输出流打印结果。

#### 相同参数再次调用

* 当再次用相同的参数调用时会执行什么？

```python
myadd(tf.constant('hello'), tf.constant('world'))
"""
0
1
2
<tf.Tensor: id=132, shape=(), dtype=string, numpy=b'helloworld'>
"""
```

* 可以看到，这里仅仅执行了上面的计算图，而没有打印 "tracing"。

#### 不同参数再次调用

* 当再次用不同的参数调用时会执行什么？

```python
myadd(tf.constant(1), tf.constant(2))
"""
tracing
0
1
2
<tf.Tensor: id=180, shape=(), dtype=int32, numpy=3>
"""
```

* 由于输入数据发生改变，所以 Tensorflow 会重新创建计算图，再执行计算图，因此 "tracing" 被打印出来。

#### 不是 Tensor

* 需要注意的是，如果输入参数不是 Tensor 类型，则每次都会重新创建计算图。

```python
myadd('hello', 'world')
myadd('good', 'morning')
"""
tracing
0
1
2
tracing
0
1
2
"""
```



### 三条规范的重新理解

* 被 @tf.function 修饰的函数应尽量**使用 Tensorflow 中的函数**而不是 Python 中的其他函数。
  * 解释：Python 中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，普通 Python 函数是无法嵌入到静态计算图中的，所以 在计算图构建好之后再次调用的时候，这些 Python 函数并没有被计算，而 TensorFlow 中的函数则可以嵌入到计算图中。使用普通的 Python 函数会导致 被 @tf.function 修饰前【eager执行】和被 @tf.function 修饰后【静态图执行】的输出不一致。
* 避免在 @tf.function 修饰的函数内部**定义 tf.Variable**。
  * 解释：如果函数内部定义了 tf.Variable, 那么在【eager执行】时，这种创建 tf.Variable 的行为在每次函数调用时候都会发生。但是在【静态图执行】时，这种创建 tf.Variable 的行为只会发生在第一步跟踪 Python 代码逻辑创建计算图时，这会导致被@tf.function修饰前【eager执行】和被 @tf.function 修饰后【静态图执行】的输出不一致。实际上，TensorFlow在这种情况下**一般会报错**。
* 被 @tf.function 修饰的函数不可修改该函数**外部的 Python 列表或字典**等结构类型变量。
  * 解释：静态计算图是被编译成 C++ 代码在 TensorFlow 内核中执行的。Python 中的列表和字典等数据结构变量是无法嵌入到计算图中，它们仅仅能够在创建计算图时被读取，在执行计算图时是无法修改 Python 中的列表或字典这样的数据结构变量的。



## @tf.function 的封装

* 由于必须避免在 @tf.function 修饰函数的内部定义 tf.Variable，那么必须在外部定义；
* 但是在外部定义 tf.Variable 的话，会显得这个函数有**外部变量依赖**，封装不够完美。
* 下面我们来解决这个问题。

### Experiment 3：子类封装

* 首先定义一个基本的函数：

```python
x = tf.Variable(1.0, dtype=tf.float32)

@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)]) # 设定输入格式：标量，float32类型
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return x

add_print(tf.constant(3.0)) # 4
# ! add_print(tf.constant(3)) # 输入不符合张量签名的参数将报错
```

* 下面利用 tf.Module 子类化将其封装：

```python
class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:
            self.x = tf.Variable(init_value, dtype=tf.float32, trainable=True)
            
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def add_print(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return self.x
```

* 最后执行该封装后的程序：

```python
demo = DemoModule(init_value=tf.constant(1.0))
result = demo.add_print(tf.constant(5.0)) # 6
```

#### 额外用法

* 除了执行该函数以外，我们简单介绍一下该子类化的额外用法：
* 查看模块中的全部变量和可训练变量：

```python
print(demo.variables)
# (<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
print(demo.trainable_variables)
# (<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
```

* 查看模块中的子模块：

```python
demo.submodules # () 这里没有子模块
```

#### 保存和加载模型（下一章详细介绍参数和方法）

* 保存模型：

```python
tf.saved_model.save(demo, './data/', signatures = {"serving_default":demo.add_print})
```

* 加载模型：

```python
demo2 = tf.saved_model.load("./data/")
demo2.add_print(tf.constant(5.0))
```

* 查看模型文件相关信息：

```python
!saved_model_cli show --dir ./data/ --all
# 输出略
```



## AutoGraph 构建实例

### Experiment 4：AutoGraph 构建模型

* 首先创建一个子类化模型，通过 @tf.function 形成 AutoGraph

```python
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)
    
    @tf.function(input_signature=[tf.TensorSpec([None,32], tf.float32)])
    def call(self, inputs):
        # 定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        return self.dense_2(x)
```

* 准备数据：

```python
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Prepare the training dataset.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
```

* 自定义模型训练，和前面一章的没什么区别：

```python
model = MyModel(num_classes=10)
epochs = 3
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # 遍历数据集的batch_size
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # 每200 batches打印一次.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))
```

```python
Start of epoch 0
Training loss (for one batch) at step 0: 43.622581481933594
Seen so far: 64 samples
Start of epoch 1
Training loss (for one batch) at step 0: 33.851985931396484
Seen so far: 64 samples
Start of epoch 2
Training loss (for one batch) at step 0: 38.966827392578125
Seen so far: 64 samples
```

* 静态图形式的模型保存：

```python
tf.saved_model.save(model,'my_saved_model')
# INFO:tensorflow:Assets written to: my_saved_model\assets
```





* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.26

