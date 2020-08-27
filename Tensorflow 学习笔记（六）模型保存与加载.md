# Tensorflow 学习笔记（六）模型保存与加载

* 本笔记将介绍模型的保存和加载方法，分别针对 Keras 模型和自定义模型来进行。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：5 Model Saving and Loading 记事本文件。



## 方法综述

### 方法一

* 使用场合：仅保存模型权重，不保存模型结构等信息。
* 格式：HDF5 格式（.h5）
* 保存方法：model.save_weights("weights_name.h5")
* 加载方法：model.load_weights("weights_name.h5")
* 注意：该方法需要有原模型代码支撑才能正确加载。不适用于模型的部署。

### 方法二

* 使用场合：保存整个模型，但不保存优化器。
* 格式：pb 格式（.pb），权重保存在 variables 文件夹中。
* 保存方法：model.save("folder_name", save_format='tf')
* 加载方法：new_model = tf.keras.models.load_model("folder_name")
* 注意 1：该方法保存的是一个文件夹，文件夹中包括 assets 文件夹、保存权重的 variables 文件夹、saved_model.pb 文件。
* 注意 2：自定义模型不能使用该方法保存。

### 方法三

* 使用场合：保存整个模型的所有部分。
* 格式：HDF5 格式（.h5）
* 保存方法：model.save("model_name.h5")
* 加载方法：new_model = tf.keras.models.load_model("model_name.h5")
* 注意：语法和方法二基本一致，但是能保存更完整的信息。

### 方法四

* 使用场合：保存整个模型，但不保存优化器。（TF2.0 推荐方法）
* 格式：pb 格式（.pb），权重保存在 variables 文件夹中。
* 保存方法：tf.saved_model.save(model, "folder_name")
* 加载方法：restored_saved_model = tf.saved_model.load("folder_name")
* 注意：该方法可以通过输入签名中的数据来达到预测的目的，详见后文实例。



## Keras 版本模型保存与加载

### Experiment 1：Keras 版本模型保存与加载实例

#### 构建模型

* 首先构建一个基本的 Keras 模型：

```python
import numpy as np
import tensorflow as tf
x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000, ))
x_val = np.random.random((200, 32))
y_val = np.random.randint(10, size=(200, ))
x_test = np.random.random((200, 32))
y_test = np.random.randint(10, size=(200, ))
```

```python
def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(32,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model
```

```python
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))
# 输出略
```

```python
model.summary() # 输出模型的基本信息，略
```

#### 方法一

```python
model.save_weights("1_weight.h5") # 保存
model.load_weights("1_weight.h5") # 加载
model.predict(x_test) # 预测
```

#### 方法二

```python
model.save('2_keras_model', save_format='tf') # 保存
new_model = tf.keras.models.load_model('2_keras_model') # 加载
new_model.predict(x_test) # 预测
```

#### 方法三

```python
model.save('3_keras_model.h5') # 保存
new_model = tf.keras.models.load_model('3_keras_model.h5') # 加载
new_model.predict(x_test) # 预测
```

#### 方法四

```python
tf.saved_model.save(model, '4_keras_model')
restored_saved_model = tf.saved_model.load('4_keras_model')
f = restored_saved_model.signatures["serving_default"] # 加载签名为f
```

* 查看签名信息：

```python
!saved_model_cli show --dir 4_keras_model --all
```

* 用测试集预测：

```python
f(digits = tf.constant(x_test.tolist()) ) # 喂入digits量来开始预测
```



## 自定义模型保存与加载

### Experiment 2：自定义模型的保存与加载实例

#### 构建模型

* 首先还是创建一个自定义模型。

```python
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)
    
    @tf.function(input_signature=[tf.TensorSpec([None,32], tf.float32,name='digits')])
    def call(self, inputs):
        #定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        return self.dense_2(x)
```

```python
x_train = np.random.random((1000, 32))
y_train = np.random.random((1000, 10))
x_val = np.random.random((200, 32))
y_val = np.random.random((200, 10))
x_test = np.random.random((200, 32))
y_test = np.random.random((200, 10))

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

        # 更新训练集的metrics
        train_acc_metric(y_batch_train, logits)

        # 每200 batches打印一次.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))

    # 在每个epoch结束时显示metrics。
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))
    # 在每个epoch结束时重置训练指标
    train_acc_metric.reset_states()

    # 在每个epoch结束时运行一个验证集。
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val)
        # 更新验证集merics
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: %s' % (float(val_acc),))
```

#### 方法一

```python
model.save_weights("1_weight.h5") # 保存
model.load_weights("1_weight.h5") # 加载
model.predict(x_test) # 预测
```

#### 方法二

* 请注意，该方法不支持子类模型的保存，因此不能使用！

```python
model.save('2_selfdefined_model', save_format='tf') # 保存
new_model = tf.keras.models.load_model('2_selfdefined_model') # 加载
new_model.predict(x_test) # 预测
```

#### 方法三

```python
# ! model.save('3_selfdefined_model.h5') # 保存
# ! new_model = tf.keras.models.load_model('3_selfdefined_model.h5') # 加载
# ! new_model.predict(x_test) # 预测
```

#### 方法四

```python
tf.saved_model.save(model, '4_selfdefined_model')
restored_saved_model = tf.saved_model.load('4_selfdefined_model')
f = restored_saved_model.signatures["serving_default"] # 加载签名为f
```

* 查看签名信息：

```python
!saved_model_cli show --dir 4_selfdefined_model --all
```

* 用测试集预测：

```python
f(digits = tf.constant(x_test.tolist()) ) # 喂入digits量来开始预测
```



#### 小注意点

* 方法一也可以用于 checkpoints 的创建：

```python
model.save_weights('./checkpoints/mannal_checkpoint')
model.load_weights('./checkpoints/mannal_checkpoint')
model.predict(x_test)
```



* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.26

