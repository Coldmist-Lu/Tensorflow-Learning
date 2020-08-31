# Tensorflow 学习笔记（九）评估函数

* 本笔记将介绍模型训练常用的评估函数，自定义评估函数的方法，最后给出一个完整的训练案例。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：8 Metric Function 记事本文件。



## 常用评估函数

* 本章的内容从格式上讲和损失函数是几乎一致的，我们简单的介绍一下，再给出几个实例。

### API

* 系统内置的损失函数被放在 tf.keras.metrics 库中（TF 1.x 版本放在 tf.metrics 中），可以从下面的网站中获得 API：
* https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/metrics



### 常见评估函数举例

#### 用于回归的评估函数

* tf.keras.metrics.MeanSquaredError：平方差误差，简写为 MSE；
* tf.keras.metrics.MeanAbsoluteError：绝对值误差，简写为 MAE；
* tf.keras.metrics.MeanAbsolutePercentageError：平均百分比误差，简写为 MAPE，函数形式为 mape；
* tf.keras.metrics.RootMeanSquaredError：均方根误差。

#### 用于分类的评估函数

* tf.keras.metrics.Accuracy：准确率，用于分类；
  * 可用字符串 "Accuracy" 表示，Accuracy = (TP + TN) / (TP + TN + FP + FN)；
  * 要求 y_true 和 y_pred 都为类别序号编码；
* tf.keras.metrics.AUC：ROC 曲线（TPR vs FPR）下的面积，用于二分类；
  * 直观解释：随机抽取一个正样本和一个负样本，正样本的预测值大于负样本的概率；
* tf.keras.metrics.Precision：精确率，用于二分类；
  * Precision = TP / (TP + FP)；
* tf.keras.metrics.Recall：召回率，用于二分类；
  * Recall = TP / (TP + FN)；
* tf.keras.metrics.TopKCategoricalAccuracy：多分类 TopK 准确率；
  * 要求 y_true(label) 为 onehot 编码形式；
* tf.keras.metrics.CategoricalAccuracy：分类准确率；
  * 与 Accuracy 含义相同，要求 y_true(label) 为 onehot 编码形式；
* tf.keras.metrics.SparseCategoricalAccuracy：稀疏分类准确率；
  * 与 Accuracy 含义相同，要求 y_true(label) 为序号编码形式。



### 类实现和函数实现

* 评估函数也存在这个问题，如：
* tf.keras.metrics.BinaryAccuracy 是类实现；
* tf.keras.metrics.binary_accuracy 是函数实现。



### Experiment 1：评估函数的调用方法

* 评估函数的使用需要用到一下几个方法：
* metrics.update_state()：更新评估函数的数据，可简写为 metrics()；
* metrics.result.numpy()：以 numpy 形式展示结果；
* metrics.reset_states()：重置函数（每一轮结束后都必须重置）

```python
m = tf.keras.metrics.Accuracy()
m.update_state([1, 2, 3, 4], [0, 2, 3, 4])
# 上面一行可以写成：
# m([1, 2, 3, 4], [0, 2, 3, 4])
print('Final result: ', m.result().numpy())
m.update_state([1, 2, 3, 4], [0, 2, 3, 1])
print('Final result: ', m.result().numpy())
m.reset_states() # 重置
```





## 自定义评估函数

* 自定义评估函数大部分都基于类进行实现，使用的基类是 tf.keras.metrics.Metric。

### 基本用法

* 自定义评估指标需要继承 tf.keras.metrics.Metric 类，并重写 init、update_state 和 result 三个方法：
  * init：所有状态变量都应通过 self.add_weight() 在此方法中创建；
  * update_state：对状态变量进行所有更新；
  * result：根据状态变量计算并返回指标值。



### Experiment 2：自定义评估函数使用实例

* 本节将使用两个实例来说明自定义评估函数的写法。

#### 稀疏多分类准确度

```python
class SparseCategoricalAccuracy_(tf.keras.metrics.Metric): # 区分名字
    def __init__(self, name='SparseCategoricalAccuracy', **kwargs):
        super(SparseCategoricalAccuracy_, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))
        
    def result(self):
        return self.count / self.total
    
    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)
```

```python
s = SparseCategoricalAccuracy_()
s.update_state(tf.constant([2, 1]), tf.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]))
print('Final result: ', s.result().numpy())
```

```python
# 与原函数比较
m = tf.keras.metrics.SparseCategoricalAccuracy()
m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
print('Final result: ', m.result().numpy())
```

#### 多分类真值数量

```python
class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))
        
    def result(self):
        return self.true_positives
    
    def reset_states(self):
        self.true_positives.assign(0.)
```

```python
y_pred = tf.nn.softmax(tf.random.uniform((4,3)))
tf.argmax(y_pred,axis=-1)
```

```python
y_true = tf.constant([2, 0, 0, 0])
```

```python
m = CategoricalTruePositives()
m.update_state(y_true, y_pred)
print('Final result:', m.result().numpy())
m.reset_states()
```





## 自定义评估函数的完整实例

### Experiment 3：自定义模型下使用自定义评估函数

#### 数据准备

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

%matplotlib inline
```

```python
mnist = np.load("../../Dataset/mnist.npz")
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0
```

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

```python
# 图片数据增加一个维度
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 建立数据集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

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

#### 自定义评估函数

```python
class CatgoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CatgoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred,axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0.)
```

```python
model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy() # 损失函数
optimizer = tf.keras.optimizers.Adam() # 优化器

#评估函数
train_loss = tf.keras.metrics.Mean(name='train_loss') # loss
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy') # 准确率
train_tp = CatgoricalTruePositives(name="train_tp") # 返回正确的个数

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_tp = CatgoricalTruePositives(name='test_tp')
```

```python
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #评估函数的结果
    train_loss(loss)
    train_accuracy(labels, predictions)
    train_tp(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_tp(labels, predictions)
```

#### 训练模型

```python
EPOCHS = 5
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_tp.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_tp.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, TP: {},Test Loss: {}, Test Accuracy: {}, Test TP:{}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          train_tp.result(),
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          test_tp.result()))
```





### Experiment 4：Keras 模型下使用自定义评估函数

```python
mnist = np.load("../../Dataset/mnist.npz")
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_train = tf.one_hot(y_train,depth=10)
y_test = tf.one_hot(y_test,depth=10)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(100).batch(32)
```

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

```python
class CatgoricalTruePositives_(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CatgoricalTruePositives_, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred,axis=-1)
        y_true = tf.argmax(y_true,axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0.)
```

```python
model = MyModel()
model.compile(optimizer = tf.keras.optimizers.Adam(0.001), #优化器
              loss =  tf.keras.losses.CategoricalCrossentropy(), #损失函数
              metrics = [tf.keras.metrics.CategoricalAccuracy(),
                         CatgoricalTruePositives_(), 
                        ] #评估函数
             ) 
```

```python
model.fit(train_ds, epochs=5, validation_data=test_ds)
```



* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.31