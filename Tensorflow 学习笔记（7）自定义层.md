# Tensorflow 学习笔记（七）自定义层

* 本笔记将介绍网络层的自定义方法，详细介绍其构造方法、注意事项，最后给出一个完整的训练案例
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：5 Model Saving and Loading 记事本文件。



## 自定义层

* 自定义层需要用到 Layer 结构，通过扩展 tf.keras.layers.Layers 类实现：
  * init 初始化魔术方法：进行所有与输入无关的初始化，定义相关的层；
  * build 方法：知道输入张量的形状并进行其余的初始化。（build 方法可有可无，变量也可以在 init 中创建；
  * call 方法：进行前向传播。
* 自定义层与自定义模型的联系与区别：
  * 自定义层使用的是 tf.keras.layers.Layer 库，自定义模型使用的是 tf.keras.Model 库；
  * 两者都通过继承的思路来使用，即通过继承 tf.keras.layers.Layer 编写自己的层，通过继承 tf.keras.Model 编写自己的类；
  * 从库本身来说，tf.keras 中的模型和层都是继承 tf.Module 实现的，tf.Module 是一个轻量级的状态容器，可以收集变量，用来建模，配合 tf.GradientTape 使用，但通常直接从 Layer 和 Model 继承即可。
  * tf.keras.Model 是继承 tf.keras.layers.Layer 实现的。

### Experiment 1：自定义层初探

* 上面说的联系，我们先验证一下：

```python
print(issubclass(tf.keras.Model,tf.Module)) # True
print(issubclass(tf.keras.layers.Layer,tf.Module)) # True
print(issubclass(tf.keras.Model,tf.keras.layers.Layer)) # True
```

#### 构建模型

* 本节我们建立一个简单的线性模型，下面是线性模型的数学表达式：
  * $y = x \cdot w + b$
  * 假设$x \in R^{(2,2)}$, $w \in R^{(2,4)} $ , $b \in R^{(4)} $
    返回 $y \in R^{(2,4)}$
* 下面的代码实现了这一线性层：

```python
class Linear(tf.keras.layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__() #
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

#### 应用模型

* 使用这一线性层：

```python
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
```

#### 结果与参数

* 展示上面的 y 结果：

```python
print(y)
"""
tf.Tensor(
[[-0.00253096  0.02058197 -0.14340366  0.06595445]
 [-0.00253096  0.02058197 -0.14340366  0.06595445]], shape=(2, 4), dtype=float32)
"""
```

* 可训练变量：

```python
linear_layer.trainable_variables
"""
[<tf.Variable 'Variable:0' shape=(2, 4) dtype=float32, numpy=
 array([[-0.03913387,  0.02152313, -0.06586333,  0.0331893 ],
        [ 0.03660291, -0.00094116, -0.07754033,  0.03276515]],
       dtype=float32)>,
 <tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
"""
```

* w 参数：

```python
linear_layer.w
"""
<tf.Variable 'Variable:0' shape=(2, 4) dtype=float32, numpy=
array([[-0.03913387,  0.02152313, -0.06586333,  0.0331893 ],
       [ 0.03660291, -0.00094116, -0.07754033,  0.03276515]],
      dtype=float32)>
"""
```

* b 参数：

```python
linear_layer.b
"""
<tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>
"""
```





## 自定义层的构造方法

### 三种构造法简介

* 本节将介绍三种自定义层的构造方法，下面具体介绍：
* 第一种方法是基本构造法，即上一节实验中使用的方法，这一方法有如下特点：
  * 在 init 方法中创建变量；
  * 创建变量需要先进行变量初始化（initializer）；然后再建立变量。
* 第二种方法采用 add_weight 方法添加权重：
  * 这样避免了先初始化再建立变量的问题，通过 add_weight 方法增加 initializer 参数即可实现初始化。
  * 相对比第一种方法更简洁，容易理解。
* 第三种方法将变量创建的任务交给 build 函数来完成。

### 不可训练参数

* 有时我们希望添加一些参数，而这些参数不参与到训练中去（固定），那么我们可以在 add_weight 方法中的 trainable 参数设置为 False，那么这些参数就不会参与训练了。
* 自定义层的不可训练参数也是可以查看的，需要查看其 non_trainable_weights 成员变量。



### Experiment 2：自定义层的构造实例

* 本文将使用鸢尾花数据集进行线性回归的分析。仅仅用于输入产生输出，从而介绍自定义层的用法，这里我们不进行训练。

#### 模型

* 模型选择：$y = x \cdot w +b $
* 假设 $x \in R^{(150,4)}$, $w \in R^{(4,1)} $ , $b \in R^{(4)} $
  返回 $y \in R^{(150,1)}$

#### 数据集导入和预处理

* 下面的代码导入了鸢尾花数据集，并进行分割：

```python
from sklearn import datasets

iris = datasets.load_iris()

data = iris.data
target = iris.target
```

```python
data.shape # (150, 4)
target.shape # (150, )
```

#### 方法一

```python
class Linear(tf.keras.layers.Layer):

    def __init__(self, units=1, input_dim=4):
        super(Linear, self).__init__() # 先调用基类初始化函数
        w_init = tf.random_normal_initializer() # 进行变量随机初始化
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'), 
                             trainable=True)
        b_init = tf.zeros_initializer() # 进行变量全零初始化
        self.b = tf.Variable(initial_value=b_init(shape=(units,),dtype='float32'), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

```python
x = tf.constant(data) # (150,4)
linear_layer = Linear(units = 1, input_dim=4) # ()
y = linear_layer(x)
print(y.shape) # (150,1)
```

```python
print(y)
```

#### 方法二

```python
class Linear(tf.keras.layers.Layer):

    def __init__(self, units=1, input_dim=4):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), # 增加权重
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,), # 增加权重
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

```python
x = tf.constant(data)
linear_layer = Linear(units = 1, input_dim=4)
y = linear_layer(x)
print(y.shape)
```

#### 方法三

```python
class Linear(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape): # 放到build方法中去
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear,self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

```python
x = tf.constant(data) # (150,4)
linear_layer = Linear(units = 1)
y = linear_layer(x)
print(y.shape)
```

#### 查看不可训练参数

* 将刚刚的线性模型的 b 参数设为不可训练，如下：

```python
class Linear(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
        super(Linear,self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

```python
x = tf.constant(data)
linear_layer = Linear(units = 1)
y = linear_layer(x)
print(y.shape)
```

* 查看所有权重、不可训练的权重、可训练权重：

```python
print('weight:', linear_layer.weights)
print('non-trainable weight:', linear_layer.non_trainable_weights)
print('trainable weight:', linear_layer.trainable_weights)
# 输出略
```





## 自定义层的注意事项

### get_config 问题

* 若在模型保存（model.save）产生下列报错，则是由于自定义网络层时 get_config 没有重写产生：

```python
NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
```

* 解决方案：get_config 的作用是获取该层的参数配置，以便模型保存时使用。看传入 init 接口的配置参数，在 get_config 内将他们转为字典键值并且返回使用。
* 例子：

```python
def get_config(self):
    config = super(Linear, self).get_config()
    config.update({'units': self.units})
    return config
```



### name 属性问题

* 若在模型保存（model.save）产生下列报错，则可能是自定义层的 build 中的 add_weight 中 name 属性没写：

```python
RuntimeError: Unable to create link (name already exists)
```



### 模型加载出错

* 自定义层建立并有效保存后，希望使用 tf.keras.models.load_model 进行模型加载时，可能会报如下错误：

```python
ValueError: Unknown layer: MyDense
```

* 解决方案：先建立一个字典，该字典的键是自定义网络层时设定该层的名字，值为自定义网络层的类名（一般是一致的）。然后在 tf.keras.models.load_model 传入 custom_objects 告知如何解析重建自定义网络层：
* 例如：

```python
_custom_objects = {
    "MyDense": MyDense,
}
tf.keras.models.load_model("model_name.h5", custom_objects=_custom_objects)
```



### 自定义层命名

* 当我们定义的自定义层名字与默认的 tf.keras 网络层一样时，可能会报出一些奇怪错误，因为重名。



### 初始化时传入 **kwargs

* 当实现自定义网络层时，最好在 init 方法中传入可变参数 **kwargs，因为有时需要对所有构成该模型的网络层进行统一的传参。

```python
def __init__(self, ... , **kwargs):
    ...
    super(MyDense, self).__init__(**kwargs)
```





## 完整案例

* 下面给出自定义层创建、训练、保存、加载的完整案例，请注意上面的注意事项在这一案例中是如何完成的。

### Experiment 3：自定义层的完整训练实例

#### 构建自定义层

```python
class MyDense(tf.keras.layers.Layer): # 注意点四：不能使用与内置层相同的名称
    def __init__(self, units=32, **kwargs): # 添加**kwargs
        self.units = units
        super(MyDense, self).__init__(**kwargs)

    # build方法一般定义Layer需要被训练的参数。    
    def build(self, input_shape): 
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='w') # 必须对可训练层设置名字
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='b')
        super(MyDense,self).build(input_shape) # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。    
    def call(self, inputs): 
        return tf.matmul(inputs, self.w) + self.b

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):  # 增设get_config方法
        config = super(MyDense, self).get_config()
        config.update({'units': self.units})
        return config
```

#### 导入数据

```python
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
labels = iris.target
```

#### 构建模型

```python
inputs = tf.keras.Input(shape=(4,))  
x = MyDense(units=16)(inputs) 
x = tf.nn.tanh(x) 
x = MyDense(units=3)(x) # 0,1,2
predictions = tf.nn.softmax(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
```

#### 打乱数据

```python
data = np.concatenate((data,labels.reshape(150,1)),axis=-1)
np.random.shuffle(data)
labels = data[:,-1]
data = data[:,:4]
```

#### 编译并训练

```python
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# keras
model.fit(data, labels, batch_size=32, epochs=100,shuffle=True)
```

```python
model.summary()
```

#### 保存模型

```python
model.save('keras_model_tf_version.h5')
```

#### 加载模型

```python
_custom_objects = {
    "MyDense" :  MyDense,
}
new_model = tf.keras.models.load_model("keras_model_tf_version.h5",custom_objects=_custom_objects)
```

#### 预测模型

```python
y_pred = new_model.predict(data)
np.argmax(y_pred,axis=1) # 输出略
labels # 输出略
```



* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.28