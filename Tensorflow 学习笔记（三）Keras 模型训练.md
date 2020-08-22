# Tensorflow 学习笔记（三）Keras 模型训练

* 本笔记将介绍如何在 Keras 模型下构建和训练网络模型。



## Keras 模型构建训练的详细过程

* Keras 训练模型主要分为下面 4 个步骤：
  * 构建模型：顺序模型、函数式模型、子类模型；
  * 模型训练：model.fit()
  * 模型验证：model.evaluate()
  * 模型预测：model.predict()
* 下面的实例将详细介绍 Keras 模型的技术细节。



### Experiment 1：Keras 模型的技术细节

#### 函数式模型构建

* 构建一个简单的模型：

```python
inputs = tf.keras.Input(shape=(32, )) # batch_size=32, 数据维度32
x = tf.keras.layers.Dense(64, activation='relu')(inputs) # 64个神经元
x = tf.keras.layers.Dense(64, activation='relu')(x) # 64个神经元
predictions = tf.keras.layers.Dense(10)(x) # 输出是10类
```

* 根据函数式模型思想，需要调用 tf.keras.Model 方法进行构建，需要指定：
  * inputs：模型输入；
  * outputs：模型输出。

```python
model = tf.keras.Model(inputs=inputs, outputs=predictions)
```

#### 损失函数、优化器、指标

* 编译模型，需要制定：
  * 损失函数（loss）；
  * 优化器（optimizer）；
  * 指标（metrics）。

```python
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), #优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #损失函数
              metrics=['accuracy']) #评估函数
```

* 通常不必从头开始创建自己的损失、指标或优化函数，所需的可能是 Keras API 的一部分：
  * 从下面的网址可看到：https://tensorflow.google.cn/guide/keras/train_and_evaluate
* 下面介绍一些常见优化器、损失和指标：
  * 优化器：SGD()、RMSprop()、Adam()；
  * 损失：MeanSquaredError()、KLDivergence()；CosineSimilarity()；
  * 指标：AUC()、Precision()、Recall().
* 如果想用上述的默认设置，那么在很多情况下，可以通过字符串标识符指定优化器、损失和指标：

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

#### 构建数据集

* 构建数据集：
  * 分别构建训练集、验证集、测试集：

```python
import numpy as np
x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000, ))

x_val = np.random.random((200, 32))
y_val = np.random.randint(10, size=(200, ))

x_test = np.random.random((200, 32))
y_test = np.random.randint(10, size=(200, ))
```

#### 模型训练

* 通过将数据切成大小为 "batch_size" 的 "批" 来训练模型，并针对给定的 "epoch" 重复遍历整个数据集：

```python
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))
"""
Train on 1000 samples, validate on 200 samples
Epoch 1/5
1000/1000 [==============================] - 4s 4ms/sample - loss: 2.3164 - accuracy: 0.1050 - val_loss: 2.3288 - val_accuracy: 0.0750
Epoch 2/5
1000/1000 [==============================] - 0s 157us/sample - loss: 2.2923 - accuracy: 0.1290 - val_loss: 2.3278 - val_accuracy: 0.0950
Epoch 3/5
1000/1000 [==============================] - 0s 141us/sample - loss: 2.2830 - accuracy: 0.1410 - val_loss: 2.3300 - val_accuracy: 0.0700
Epoch 4/5
1000/1000 [==============================] - 0s 159us/sample - loss: 2.2740 - accuracy: 0.1480 - val_loss: 2.3285 - val_accuracy: 0.0650
Epoch 5/5
1000/1000 [==============================] - 0s 143us/sample - loss: 2.2608 - accuracy: 0.1590 - val_loss: 2.3339 - val_accuracy: 0.0900
<tensorflow.python.keras.callbacks.History at 0x2036100ee48>
"""
```

#### 验证集的自动划分

* 在前面的例子中，我们使用 validation_data 参数将 Numpy 数组的元组 (x_val, y_val) 传递给模型，在每个时期结束时评估验证损失和验证指标。
* 还有一种策略是：参数 validation_split 允许您自动保留部分训练数据以供验证。参数值代表要保留用于验证的数据的一部分，因此应将其设置为大于 0 且小于 1 的数字。例如，validation_split=0.2 表示“使用20％的数据进行验证”，而 validation_split=0.6 表示“使用60％的数据用于验证”。

* 验证的计算方法是在进行任何改组之前，对fit调用接收到的数组进行最后x％的采样。

* 注意，validation_split 只能在使用 Numpy 数据进行训练时使用。

```python
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
"""
Train on 800 samples, validate on 200 samples
800/800 [==============================] - 0s 173us/sample - loss: 2.2485 - accuracy: 0.1475 - val_loss: 2.2530 - val_accuracy: 0.1350
<tensorflow.python.keras.callbacks.History at 0x20382497908>
"""
```

#### 模型验证

* 返回 test loss 和 metrics

```python
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)
"""
# Evaluate on test data
200/1 [==...==] - 0s 85us/sample - loss: 2.3017 - accuracy: 0.1100
test loss, test acc: [2.3221388721466063, 0.11]

# Generate predictions for 3 samples
predictions shape: (3, 10)
"""
```





## 样本加权与类别加权

### 权重的作用

#### 传递权重的基本方法

* 在使用时将**样本权重**或**类权重**传递给模型 fit：
* 从 Numpy 数据进行训练时：通过 sample_weight 和 class_weight 参数。
* 从数据集训练时：通过使数据集返回一个元组 (input_batch, target_batch, sample_weight_batch)。

#### 权重的构成与含义

*  “样本权重” 数组是一个数字数组，用于指定批次中每个样本在计算总损失时应具有的权重。它通常用于**不平衡的分类问题**中（这种想法是为很少见的班级赋予更多的权重）。当所使用的权重为1和0时，该数组可用作损失函数的**掩码**（完全丢弃某些样本对总损失的贡献）。
* “类别权重” 字典是同一概念的一个更具体的实例：它将类别索引**映射到应该用于属于该类别的样本**的样本权重。例如，如果在数据中类“ 0”的表示量比类“ 1”的表示量少两倍，则可以使用 class_weight={0: 1., 1: 0.5}。



### Experiment 2：权重增设实例

#### 构建模型

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

#### 类别加权

* 下面的 Numpy 示例，使用类权重来更加**重视第5类的正确分类**。

```python
import numpy as np

#类别加权
class_weight = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
                5: 2.,
                6: 1., 7: 1., 8: 1., 9: 1.}

print('Fit with class weight')
model = get_compiled_model()
model.fit(x_train, y_train, # 这里的x_train和y_train使用的是上面例子的数据集
          class_weight=class_weight,
          batch_size=64,
          epochs=4)
```

```python
"""
Fit with class weight
Train on 1000 samples
Epoch 1/4
1000/1000 [==============================] - 5s 5ms/sample - loss: 2.4946 - sparse_categorical_accuracy: 0.0940
Epoch 2/4
1000/1000 [==============================] - 0s 115us/sample - loss: 2.4690 - sparse_categorical_accuracy: 0.0820
Epoch 3/4
1000/1000 [==============================] - 0s 118us/sample - loss: 2.4581 - sparse_categorical_accuracy: 0.0820s - loss: 2.4543 - sparse_categorical_accuracy: 0.085
Epoch 4/4
1000/1000 [==============================] - 0s 108us/sample - loss: 2.4489 - sparse_categorical_accuracy: 0.0860
<tensorflow.python.keras.callbacks.History at 0x1e1929c7f28>
"""
```

#### 样本加权

* 下面我们用相同的方式对样本加权：

```python
# Here's the same example using `sample_weight` instead:
sample_weight = np.ones(shape=(len(y_train),))

sample_weight[y_train == 5] = 2.

print('\nFit with sample weight')
model = get_compiled_model()
model.fit(x_train, y_train,
          sample_weight=sample_weight,
          batch_size=64,
          epochs=4)
```

```python
"""
Train on 1000 samples
Epoch 1/4
1000/1000 [==============================] - 1s 758us/sample - loss: 2.4922 - sparse_categorical_accuracy: 0.0820
Epoch 2/4
1000/1000 [==============================] - 0s 91us/sample - loss: 2.4647 - sparse_categorical_accuracy: 0.0830
Epoch 3/4
1000/1000 [==============================] - 0s 100us/sample - loss: 2.4562 - sparse_categorical_accuracy: 0.0880
Epoch 4/4
1000/1000 [==============================] - 0s 85us/sample - loss: 2.4523 - sparse_categorical_accuracy: 0.0890
<tensorflow.python.keras.callbacks.History at 0x1e1f098c518>
"""
```



## 回调函数

* Keras 中的回调是在训练期间（在某个时期开始时，在批处理结束时，在某个时期结束时等）在不同时间点调用的对象。
* 这些对象可用于实现以下行为：
  * 在训练过程中的**不同时间点进行验证**（除了内置的按时间段验证）；
  * 定期或在**超过特定精度阈值**时对模型进行检查；
  * 当训练似乎停滞不前时，**更改模型的学习率**；
  * 当训练似乎停滞不前时，**对顶层进行微调**；
  * 在训练结束或超出特定性能阈值时**发送电子邮件或即时消息通知** 等等。
* 回调要作为**列表（list）**传递给 **model.fit**：



### 常用回调函数

* **ModelCheckpoint**：定期保存模型。
* **EarlyStopping**：当训练不再 改善验证指标时，停止培训。
* **TensorBoard**：定期编写可在 TensorBoard 中可视化的模型日志（更多详细信息，请参见“可视化”部分）。
* **CSVLogger**：将损失和指标数据流式传输到 CSV 文件。



### EarlyStopping（早停机制）

* **monitor**: 被监测的数据。
* **min_delta**: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
* **patience**: 没有进步的训练轮数，在这之后训练就会被停止。
* **verbose**: 详细信息模式。
* **mode**: {auto, min, max} 其中之一。 在 min 模式中， 当被监测的数据停止下降，训练就会停止；在 max 模式中，当被监测的数据停止上升，训练就会停止；在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。

### Experiment 3：早停机制实例

* 下例展示了如何在 val_loss 不再下降时停止训练：

```python
model = get_compiled_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping( 
        monitor='val_loss',        
        min_delta=1e-2, # “不再下降”被定义为“减少不超过1e-2”
        patience=2, # “不再改善”进一步定义为“至少2个epoch”
        verbose=1)
]

model.fit(x_train, y_train,
          epochs=20,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2)
```

```python
"""
Train on 800 samples, validate on 200 samples
Epoch 1/20
800/800 [==============================] - 1s 792us/sample - loss: 2.3303 - sparse_categorical_accuracy: 0.0875 - val_loss: 2.3051 - val_sparse_categorical_accuracy: 0.1500
Epoch 2/20
800/800 [==============================] - 0s 126us/sample - loss: 2.2995 - sparse_categorical_accuracy: 0.1112 - val_loss: 2.3060 - val_sparse_categorical_accuracy: 0.1350
Epoch 3/20
800/800 [==============================] - 0s 111us/sample - loss: 2.2903 - sparse_categorical_accuracy: 0.1163 - val_loss: 2.3056 - val_sparse_categorical_accuracy: 0.1300
Epoch 00003: early stopping
"""
```



### checkpoint 模型

* 在相对较大的数据集上训练模型时，至关重要的是要定期保存模型的 checkpoint。
* 最简单的方法是使用 ModelCheckpoint 回调。

### Experiment 4：ModelCheckpoint 回调实例

```python
model = get_compiled_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='mymodel_{epoch}', # 模型保存路径
        save_best_only=True, # 当且仅当`val_loss`分数提高时，我们才会覆盖当前检查点。
        monitor='val_loss',
        save_weights_only=True, # 仅仅保存模型权重
        verbose=1)
]

model.fit(x_train, y_train,
          epochs=3,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2)
```

```python
"""
Train on 800 samples, validate on 200 samples
Epoch 1/3
768/800 [===========================>..] - ETA: 0s - loss: 2.3034 - sparse_categorical_accuracy: 0.1159
Epoch 00001: val_loss improved from inf to 2.31087, saving model to mymodel_1
800/800 [==============================] - 1s 2ms/sample - loss: 2.3033 - sparse_categorical_accuracy: 0.1175 - val_loss: 2.3109 - val_sparse_categorical_accuracy: 0.1300
Epoch 2/3
704/800 [=========================>....] - ETA: 0s - loss: 2.2837 - sparse_categorical_accuracy: 0.1449
Epoch 00002: val_loss did not improve from 2.31087
800/800 [==============================] - 0s 131us/sample - loss: 2.2813 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.3111 - val_sparse_categorical_accuracy: 0.1000
Epoch 3/3
512/800 [==================>...........] - ETA: 0s - loss: 2.2681 - sparse_categorical_accuracy: 0.1641
Epoch 00003: val_loss did not improve from 2.31087
800/800 [==============================] - 0s 141us/sample - loss: 2.2722 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.3117 - val_sparse_categorical_accuracy: 0.1000
"""
```



### Experiment 5：动态学习率调整

* 由于优化程序无法访问验证指标，因此无法使用这些计划对象来实现动态学习率计划（例如，当验证损失不再改善时降低学习率）。
* 但是，回调确实可以访问所有指标，包括验证指标！因此，可以通过使用回调来修改优化程序上的当前学习率，从而实现此模式。实际上，它是作为 ReduceLROnPlateau 回调内置的。

#### 参数

* ReduceLROnPlateau参数
  - monitor: 被监测的指标。
  - factor: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
  - patience: 没有进步的训练轮数，在这之后训练速率会被降低。
  - verbose: 整数。0：安静，1：更新信息。
  - mode: {auto, min, max} 其中之一。如果是 min 模式，学习速率会被降低如果被监测的数据已经停止下降； 在 max 模式，学习塑料会被降低如果被监测的数据已经停止上升； 在 auto 模式，方向会被从被监测的数据中自动推断出来。
  - min_delta: 衡量新的最佳阈值，仅关注重大变化。
  - cooldown: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
  - min_lr: 学习速率的下边界。

#### 程序

```python
model = get_compiled_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='mymodel_{epoch}', # 模型保存路径
        save_best_only=True, # 当且仅当`val_loss`分数提高时，才会覆盖当前检查点。
        monitor='val_loss',
        save_weights_only=True, # 仅仅保存模型权重
        verbose=1),
    
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy", 
                                         verbose=1, 
                                         mode='max', 
                                         factor=0.5, 
                                         patience=3)
]
model.fit(x_train, y_train,
          epochs=30,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2
         )
# 输出略
```



## 多输入多输出模型训练实例

### Experiment 6：多输入多输出模型

* 考虑以下模型，该模型具有形状的图像输入(32, 32, 3)（即(height, width, channels)）和形状的时间序列输入(None, 10)（即(timesteps, features)）。模型将具有根据这些输入的组合计算出的两个输出：“得分”（形状(1,)）和五类（形状(5,)）的概率分布。

### 模型构建

#### 构建模型

```python
image_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = tf.keras.Input(shape=(20, 10), name='ts_input')

x1 = tf.keras.layers.Conv2D(3, 3)(image_input)
x1 = tf.keras.layers.GlobalMaxPooling2D()(x1)


x2 = tf.keras.layers.Conv1D(3, 3)(timeseries_input)
x2 = tf.keras.layers.GlobalMaxPooling1D()(x2)

x = tf.keras.layers.concatenate([x1, x2])

score_output = tf.keras.layers.Dense(1, name='score_output')(x)
class_output = tf.keras.layers.Dense(5, name='class_output')(x)

model = tf.keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[score_output, class_output])
```

#### 绘制模型流程

* 用下面的函数，安装指定包即可画出模型的示意图：

```python
tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True,dpi=500)
```

* 示意图详见代码文件。

#### 损失函数

* 在编译时，通过将损失函数作为列表传递，我们可以为不同的输出指定不同的损失：

```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
          tf.keras.losses.CategoricalCrossentropy(from_logits=True)])
```

* 如果我们仅将单个损失函数传递给模型，则将相同的损失函数应用于每个输出，这在此处不合适。

#### 指标函数

```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
          tf.keras.losses.CategoricalCrossentropy(from_logits=True)],
    metrics=[
            [tf.keras.metrics.MeanAbsolutePercentageError(),
             tf.keras.metrics.MeanAbsoluteError()],        
            [tf.keras.metrics.CategoricalAccuracy()]
    ]
)
```

* 由于我们为输出层命名，因此我们还可以通过dict指定每个输出的损失和指标：

```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={'score_output': tf.keras.losses.MeanSquaredError(),
          'class_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
         },
    metrics={'score_output': [tf.keras.metrics.MeanAbsolutePercentageError(),
                              tf.keras.metrics.MeanAbsoluteError()],             
             'class_output': [tf.keras.metrics.CategoricalAccuracy()]}
)
```

* 如果您有两个以上的输出，我们建议使用显式名称和字典。

#### 赋予权重

* 可以使用以下参数对不同的特定于输出的损失赋予不同的权重（例如，在我们的示例中，我们可能希望通过将某类损失函数赋予更高的权重）：

```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={'score_output': tf.keras.losses.MeanSquaredError(),
          'class_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
    metrics={'score_output': [tf.keras.metrics.MeanAbsolutePercentageError(),
                              tf.keras.metrics.MeanAbsoluteError()],
             'class_output': [tf.keras.metrics.CategoricalAccuracy()]},
    loss_weights={'score_output': 2., 'class_output': 1.})
```

* 可以选择不为某些输出计算损失。如果这些输出仅用于预测而不是训练：

```python
# List loss version
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[None, tf.keras.losses.CategoricalCrossentropy(from_logits=True)])

# Or dict loss version
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={'class_output':tf.keras.losses.CategoricalCrossentropy(from_logits=True)})
```



### 完整运行

* 下面给出编译模型和运行的完整代码：

```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
          tf.keras.losses.CategoricalCrossentropy(from_logits=True)])

# Generate dummy Numpy data
import numpy as np
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets],
          batch_size=32,
          epochs=3)

# Alternatively, fit on dicts
model.fit({'img_input': img_data, 'ts_input': ts_data},
          {'score_output': score_targets, 'class_output': class_targets},
          batch_size=32,
          epochs=3)
```



* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.8.22