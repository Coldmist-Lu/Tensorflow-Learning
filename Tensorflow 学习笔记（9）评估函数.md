# Tensorflow 学习笔记（九）评估函数

* 本笔记将介绍模型训练常用的优化函数，自定义损失函数的类方法和函数方法，最后给出一个完整的训练案例。
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

#### 

