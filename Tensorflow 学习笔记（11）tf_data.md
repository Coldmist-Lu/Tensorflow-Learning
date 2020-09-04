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

#### from tensors 和 from_tensor_slices 的区别

* 基于列表构造：

```python
ds = tf.data.Dataset.from_tensor_slices([1,2,3,4])
for line in ds:
    print(line)
"""
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
"""
```

```python
ds = tf.data.Dataset.from_tensors([1,2,3,4])
for line in ds:
    print(line)
"""
tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)
"""
```

* 基于 constant + 字典 构造：

```python
t = tf.constant([[1, 2], [3, 4]])
ds = tf.data.Dataset.from_tensors(t)   # [[1, 2], [3, 4]]
for line in ds:
    print(line)    
"""
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
"""
```

```python
t = tf.constant([[1, 2], [3, 4]])
ds = tf.data.Dataset.from_tensors({"a":t,"b":t})   # [[1, 2], [3, 4]]
for line in ds:
    print(line)
    break
"""
{'a': <tf.Tensor: id=43, shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]])>, 'b': <tf.Tensor: id=44, shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]])>}
"""
```

* 基于 numpy 构造：

```python
dataset1 = tf.data.Dataset.from_tensors(np.zeros(shape=(10,5,2), dtype=np.float32))
for line in dataset1:
    print(line.shape) # (10, 5, 2)
    break
```

```python
dataset2 = tf.data.Dataset.from_tensor_slices(np.zeros(shape=(10,5,2), dtype=np.float32))
for line in dataset2:
    print(line.shape) # (5, 2)
    break
```

* 基于 numpy + 字典 构造：

```python
dataset3=tf.data.Dataset.from_tensors({"a":np.zeros(shape=(10,5,2), dtype=np.float32),
                                       "b":np.zeros(shape=(10,5,2), dtype=np.float32)})
for line in dataset3:
    print(line['a'].shape,line['b'].shape) # (10, 5, 2) (10, 5, 2)
    break
```

```python
dataset3=tf.data.Dataset.from_tensor_slices({"a":np.zeros(shape=(10,5,2), dtype=np.float32),
                                       "b":np.zeros(shape=(10,5,2), dtype=np.float32)})
for line in dataset3:
    print(line['a'].shape,line['b'].shape) # (5, 2) (5, 2)
    break
```

#### numpy 完整数据读取

```python
mnist = np.load("../../Dataset/mnist.npz")
x_train, y_train = mnist['x_train'], mnist['y_train']

x_train.shape, y_train.shape
```

```python
x_train = np.expand_dims(x_train, axis=-1) # 增加颜色通道维
x_train.shape
```

```python
mnist_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) # 构建数据集
```

```python
for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :,0])
    plt.show()
    break
```

#### pandas 数据读取

```python
df = pd.read_csv("../../Dataset/heart.csv")
df.head()
```

* 转换数据类型：

```python
df.dtypes
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
```

* 建立数据集：

```python
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))
```

#### 从 Python generator 构建数据管道

```python
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
flowers = '../../Dataset/flower_photos'
```

```python
def Gen():
    gen = img_gen.flow_from_directory(flowers)
    for (x, y) in gen:
        yield (x, y)
```

```python
ds = tf.data.Dataset.from_generator(
    Gen,
    output_types=(tf.float32, tf.float32)
)
```

```python
for image, label in ds:
    print(image.shape, label.shape)
    break
```





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

```python
feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}
```

```python
def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']
```

```python
batch_size = 32
train_dataset = tf.data.TFRecordDataset("../../Dataset/sub_train.tfrecords")
train_dataset = train_dataset.map(_parse_example)
for image, label in train_dataset:
    print(image.shape, label)
    break
```





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

```python
titanic_lines = tf.data.TextLineDataset(["../../Dataset/titanic_dataset/train.csv", "../../Dataset/titanic_dataset/eval.csv"])
```

```python
def data_func(line):
    line = tf.strings.split(line, sep=',')
    return line
```

```python
titanic_data = titanic_lines.skip(1).map(data_func)
```

```python
for line in titanic_data:
    print(line)
    break
```





* Written by：Sirius. Lu
* Reference：深度之眼《Tensorflow 框架训练营》
* 2020.9.3

