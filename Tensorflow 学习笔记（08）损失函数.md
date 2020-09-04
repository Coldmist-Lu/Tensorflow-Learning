# Tensorflow 学习笔记（八）损失函数

* 本笔记将介绍模型训练常用的损失函数，自定义损失函数的类方法和函数方法，最后给出一个完整的训练案例。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：7 Loss Function 记事本文件。



## 常用损失函数

### API

* 系统内置的损失函数被放在 tf.keras.losses 库中，可以从下面的网站中获得 API：
* https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/losses



### 常见损失函数举例

* 下面列举几个常用的损失函数。
  * 平方差误差损失 mean_squared_error
    * 平方差误差损失函数用于**回归**，简写为 mse，类的实现形式为 **MeanSquaredError** 和 MSE。
  * 二元交叉熵 binary_crossentropy
    * 二元交叉熵损失函数用于**二分类**，类实现形式为 **BinaryCrossentropy**。
  * 类别交叉熵 categorical_crossentropy
    * 类别交叉熵损失函数用于多分类，要求 label 为 onehot 编码，类实现形式为 CategoricalCrossentropy。
  * 稀疏类别交叉熵 sparse_categorical_crossentropy
    * 稀疏类别交叉熵用于多分类，要求 label 为序号编码形式，类实现形式为 SparseCategoricalCrossentropy。



### 类实现与函数实现

* 请注意，损失函数有**类实现**与**函数实现**两种方式，调用时没有明显的效果区别。
* 类实现：使用驼峰命名法，每个单词的首字母大写，如 BinaryCrossentropy；
* 函数实现：使用下划线命名法，单词之间使用下划线分隔，如 binary_crossentropy.



### Experiment 1：交叉熵损失函数应用实例

* 下面我们以多分类的交叉熵损失函数为例，介绍一下 tensorflow 中的损失函数调用方法。

#### 公式

* 多分类交叉熵损失函数可以用下面的表达式来刻画：

$$
L = \frac1N \sum_{i=1}^N L_i = - \frac1N \sum_{i=1}^N \sum_{c=1}^M y_{ic} \log(p_{ic})
$$

* 其中，$N$ 表示样本的数量，$M$ 表示分类数量；
* 那么，$L_i$ 表示第 $i$ 个样本的损失，$y_{ic}$ 表示真实情况第 $c$ 个分类是否为第 $i$ 个样本的真实分类，$p_{ic}$ 表示模型预测出的样本为该分类的概率。

#### numpy 版本的实现

* 如果是用 numpy 库，可以用下面的代码来实现：

```python
a = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
b = np.array([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])
print(np.average(-np.sum(a*np.log(b), axis=1)))
```

#### tensorflow 的实现

* 下面展示用 tensorflow 库实现：

```python
cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]]
)
print("Loss: ", loss.numpy())
```





## 自定义损失函数

* 自定义损失函数的类实现继承自 tf.keras.losses.Loss 类，API：
* https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/losses/Loss
* 自定义损失函数也分函数和类两种实现形式。

### Experiment 2：MSE 损失函数的三种实现

#### 类实现

* 下面的代码是用类来自定义损失函数：

```python
class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
a = tf.constant([0., 0., 1., 1.])
b = tf.constant([1., 1., 1., 0.])
mse = MeanSquaredError()
tf.print(mse(a, b))
```

#### 函数实现

* 下面的代码是用直接函数来自定义：

```python
def MeanSquaredError(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

mse1 = MeanSquaredError(a, b)
print(mse1)
```

#### 直接调用库

```python
mse = tf.keras.losses.MeanSquaredError()
loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
print(loss.numpy())
```



## 自定义损失函数的实例

### Focal loss 损失函数

* Focal loss 是一种用于图像分类的损失函数：
* 论文地址：Focal loss for Dense Object Detection
* 相关讨论：https://www.zhihu.com/question/63581984
* 这里将二分类的 Focal loss 应用于多分类，解决了类别不平衡的分类问题。

#### 公式

$$
FL(p_t) = -\sum_{c=1}^m (1-p_t)^\gamma * y_c * \log(p_t)
$$

* 其中，$m$ 表示类别总数，$y_c$ 是真实标签，$p_t$ 是预测标签，$\gamma$ 是调节因子。

#### 类实现

* 下面是一种 SparseFocalLoss 的类实现：

```python
class SparseFocalLoss(tf.keras.losses.Loss):
    
    def __init__(self, gamma=2.0, alpha=0.25, class_num=10):
        self.gamma = gamma
        self.alpha = alpha
        self.class_num = class_num
        super(SparseFocalLoss, self).__init__()
        
    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon() # 产生一个约为1e-7的浮点数
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        
        y_true = tf.one_hot(y_true, depth=self.class_num)
        y_true = tf.cast(y_true, tf.float32)
        
        loss = - y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss
```

* 下面我们对代码中没有出现过的函数进行简单介绍：
  * tf.nn.softmax() ：softmax 函数，即将预测值的输出结果转换至 0~1 的概率结果；
  * tf.keras.backend.epsilon()：产生一个极小的正浮点数，约为 1e-7，这是为了下一步保证结果不为 0；
  * tf.clip_by_value()：对数据进行截取，即将小于 epsilon 的数据替换成 epsilon，大于 1 的数据替换为 1；
  * tf.one_hot()：将标注结果转化成 one_hot 编码；
  * tf.cast()：前面张量操作的笔记介绍过，数据类型转换。

#### 函数实现

* 下面再给出一个函数的实现方法，这里我们再写一个非 sparse （稀疏）的实现，因为这些函数是自定义的，可以根据实际情况修改：

```python
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        
        y_true = tf.cast(y_true, tf.float32)
        
        loss = - y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        
        return loss
    return focal_loss_fixed
```



### Experiment 3：自定义模型下使用自定义损失函数

* 下面给出一个完整的实例，在自定义模型下使用自定义损失函数进行训练。
* 本实例使用的是 MNIST 手写识别数据集。在实战部分给出了这个数据集的文件下载地址以及各部分的执行结果。

#### 导入数据，划分训练集和测试集

```python
mnist = np.load("../../Dataset/mnist.npz") # 实际使用自己调整路径
x_train, y_train, x_test, y_test = mnist['x_train'],mnist['y_train'],mnist['x_test'],mnist['y_test']
x_test.shape
x_train, x_test = x_train / 255.0, x_test / 255.0
```

#### 简单可视化

```python
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )
 
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
```

#### 数据集变换

```python
# 图片数据增加一个维度
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 标签数据换成one hot编码
y_train = tf.one_hot(y_train,depth=10)
y_test = tf.one_hot(y_test,depth=10)

# 建立数据集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

#### 建立模型

```python
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
```

#### 自定义损失 —— 类实现

```python
class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        super(FocalLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred,axis=-1)
        epsilon = tf.keras.backend.epsilon()#1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)        
       
        y_true = tf.cast(y_true,tf.float32)
        
        loss = -  y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)        
        loss = tf.math.reduce_sum(loss,axis=1)
        return loss
```

#### 自定义损失 —— 函数实现

```python
def FocalLoss(gamma=2.0, alpha=0.25):
    
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred,axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true = tf.cast(y_true,tf.float32)

        loss = -  y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss,axis=1)
        return loss
    
    return focal_loss_fixed
```

#### 定义模型、损失函数、优化器和评估函数

```python
model = MyModel()

loss_object = FocalLoss(gamma=2.0,alpha=0.25)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
```

#### 将训练和检测的任务封装成函数

```python
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

#### 训练模型

```python
EPOCHS = 5
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
```



### Experiment 4：Keras 模型下使用自定义损失函数

* 下面给出一个完整的实例，在 Keras 模型下使用自定义损失函数。
* 本例基于 Experiment 3，对数据已经进行好了导入和预处理，用的是相同的数据。

#### 定义模型

```python
def MyKerasModel():
    inputs = tf.keras.Input(shape=(28,28,1), name='digits')
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
```

#### 编译模型

```python
model = MyModel()
model.compile(optimizer = tf.keras.optimizers.Adam(0.001), #优化器
              loss =  FocalLoss(gamma=2.0, alpha=0.25), #损失函数
              metrics = [tf.keras.metrics.CategoricalAccuracy()] #评估函数
             ) 
```

#### 训练模型

```python
model.fit(train_ds, epochs=5, validation_data=test_ds)
```



* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.30