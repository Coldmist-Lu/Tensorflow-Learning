# Tensorflow 学习笔记（十一）tf.data 简介

* 本笔记将简单介绍模型输入管道 tf.data 模块，后续笔记分模块介绍进阶用法。
* 本笔记实验代码的输出结果详见 Tensorflow2.0-in-action 仓库：9 TensorBoard 记事本文件。



## tf.data 简介

* TensorFlow 提供了 tf.data 模块，该模块是一套灵活构建数据集的 API，能快速、高效地构建数据输入，适用于数据量巨大的场景。
* tf.data 模块主要包含三个库：
  * tf.data.Dataset 类
  * tf.data.TFRecordDataset 类
  * tf.data.TextLineDataset 类
* 下面分别介绍。





## tf.data.Dataset 简介

* tf.data.Dataset 是 tf.data 模块的核心类，提供了对数据集的高层封装。
* tf.data.Dataset 由一系列的可迭代访问元素（element）组成，每个元素包含一个或多个张量。
* Dataset 可以看成是**相同类型 "元素" 的有序列表**，单个元素可以是向量，也可以是字符串、图片、tuple 或 dict。
* 比如说，对于一个由图像组成的数据集，每个元素可以是形状为 长 × 宽 × 通道数 的图片张量，也可以是由图片张量和图片标签张量组成的元组。



### 创建数据集

* 以下列举了几种 Dataset 类创建数据集（将类实例化）的方式：
* **tf.data.Dataset.from_tensors()**：创建 Dataset 对象，将输入看成是单个元素，并返回该元素的数据集（用得少，因为一般情况下我们不会只有一个数据。）
* **tf.data.Dataset.from_tensor_slices()**：创建 Dataset 对象，输入可以是一个或多个 tensor，若是多个 tensor，需要以元组或字典封装。（如果只传入一个元素，该方法会将数据的**第 0 维作为元素个数**，其余各维度分别合并成各个数据）
* **tf.data.Dataset.from_generator()**：以迭代器的方式生成数据集，在数据量较大的时候适合使用。



### 处理数据集

* 以下列举了一些数据集预处理的方法：
* **tf.data.Dataset.map(f)**：对数据集中的每个元素都应用函数 f，得到一个新的数据集（往往结合 tf.io 进行读写或解码文件，或 tf.image 进行图像处理）
* **tf.data.Dataset.shuffle(buffer_size)**：将数据集打乱（设定一个固定大小的缓冲区 buffer，取出前 buffer_size 个元素放入，并从缓冲区随机采样，采样后的数据用后续数据替换）
* **tf.data.Dataset.batch(batch_size)**：将数据集分成批次，即对每 batch_size 个元素，使用 tf.stack() 在第 0 维合并，称为一个元素。

 

### Experiment 1：数据集的创建方法







## tf.data.TFRecordDataset 简介

* TFRecordDataset 专门针对特别巨大而无法完整载入内存的数据集使用，可以先将数据集处理为 TFRecord 格式，然后进行载入。
* 当将数据集整理成 TFRecord 格式后，Tensorflow 就可以高效地读取和处理数据集，从而方便进行大规模的模型训练。



### 核心代码说明

```python
tf.data.TFRecordDataset(
    filenames, compression_type=None, buffer_size=None, num_parallel_reads=None
)
```

* filenames：tf.string 张量，值可以是一个或用列表构成的多个文件名。
* compression_type：tf.string 标量，值为 "" (不压缩)、"ZLIB" 和 "GZIP" 之一。
* buffer_size：tf.int64 标量，表示读取缓冲区的字节数。
* num_parallel_reads：tf.int64 标量，表示要并行读取的文件数。



### Experiment 2：TFRecordDataset 用法实例





## tf.data.TextLineDataset 简介

* tf.data.TextLineDataset 提供了一种从一个或多个文本文件中提取行的简单方法。
* 给定一个或多个文件名，TextLineDataset 会为这些文件的每行生成一个字符串值元素。
* 类中保存的一行就是一个元素，是 string 类型的 tensor。



### 核心代码

```python
tf.data.TextLineDataset(
    filenames, compression_type=None, buffer_size=None, num_parallel_reads=None
)
```

* 其中参数的作用和上面的一样。这里不再赘述了。



### Experiment 3：TextLineDataset 用法实例





* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.9.3

