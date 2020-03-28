# 学习RNN神经网络


## 时间序列问题

![image](https://dikers-data.s3.cn-northwest-1.amazonaws.com.cn/images/rnn.png)




## 1. 用numpy实现简单的RNN

实现一个简单的RNN， 预测字符串序列, 理解一些常用的矩阵运算， 以及前向反向传播
![image](https://pic1.zhimg.com/80/v2-4058db6817f202ddc3fc41cb3683a744_1440w.png)



## 2. 使用Tensorflow 实现RNN 预测Sin函数图像

* 理解序列问题数据处理方式， X 和y的数据格式。 
* 熟悉Tensorflow 构建网络模型的常规流程。 


## 3. 使用LSTM 对mnist 手写数字集进行分类


![image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1585420089066&di=b7aecd71e3249cf1c82c95858ad0cf90&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-78abf1f3cfb557f9e4dd2fbb9c135ecc_b.jpg)

![image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1585420330395&di=fe0f331f114848935b109a657958fd6c&imgtype=0&src=http%3A%2F%2Finews.gtimg.com%2Fnewsapp_bt%2F0%2F6847845171%2F1000)

将每行数据看做时序序列对图像进行分类。 



## 4. 实现自己的LSTM 

通过实现LSTM， 理解循环网络的内部结构， 加深理解。 

![image](https://img2018.cnblogs.com/blog/1575054/201904/1575054-20190412231515774-190310739.png)


## 5. 通过Encoder Decoder实现 简单问答系统

序列到序列的模型， 核心就是研究如何处理序列的问题。 

```
Q: 24+654
A: 678
```
程序读取问题字符串，通过和答案的对比， 发现序列里面的规律。  



## 6. 使用PyTorch实现字符级别的英语文章预测

```
The moss of the convincing it had been drawing up the people that there was nothing without this way or a single wife as he did not hear
him or that he was not seeing that she would be a court of the sound of some sound of the position, and to spartly she
could
see her and a sundroup times there was nothing this
father and as she stoop serious in the sound, was a steps of the master, a few sistersily play of his husband. The crowd had no carreated herself, and truets, and shaking up, the pases, and the moment that he was not at the marshal, and the starling the secret were stopping to be
```

## 7. 实现了中文文章的分类

* 熟悉多对一个序列问题的解决方案
* 熟悉中文文档的分词处理流程

里面使用了9类文章， 每类文章有5000片文章。 

| 分类标签 | 出现的数量 |
| :-----| ----: |
| 体育 | 5000 |
| 娱乐 | 5000 |
| 家居 | 5000 |
| 房产 | 5000 |
| 时尚 | 5000 |
| 时政 | 5000 |
| 游戏 | 5000 |
| 科技 | 5000 |
| 财经 | 5000 |


## 8. 实现英语到西班牙的翻译功能

* 熟悉多对多的序列问题
* 熟悉Attention模型

以下是一些样本数据：
```
You don't look the same.	No luces lo mismo.
You don't pay attention.	No prestás atención.
You don't smoke, do you?	No fumas, ¿verdad?
You drink too much, Tom.	Bebes demasiado, Tom.
You dropped your pencil.	Se te ha caído el lápiz.
You dropped your pencil.	Se te cayó el lápiz.
You go there without me.	Tú ve allá sin mí.
You go to school, right?	Tú vas a la escuela, ¿сierto?
You guys need new shoes.	Necesitáis zapatos nuevos.
You handled that deftly.	Manejaste eso hábilmente.
You have a lot of books.	Tú tienes muchos libros.
You have a pretty smile.	Tienes una linda sonrisa.
You have beautiful eyes.	Tienes unos ojos muy bonitos.
```

## 9. 自动生成作曲

* 熟悉声音文件的处理
* 熟悉声音序列问题的处理方式



## 10. 通过记忆网络实现问答系统

```
Daniel moved to the garden . Mary went back to the bathroom .

Q: Where is Daniel ?

A: garden
```


```
John went to the bedroom . Daniel journeyed to the kitchen . Daniel journeyed to the hallway . Mary travelled to the bathroom . John travelled to the garden . Daniel journeyed to the office . Mary moved to the bedroom . Sandra went back to the hallway .

Q: Where is Sandra ?

A: hallway

```


## 11. 通过手机传感器预测人类运动

熟悉时间序列的处理方式


对应的6个分类： 
```
    "WALKING"
    "WALKING_UPSTAIRS"
    "WALKING_DOWNSTAIRS" 
    "SITTING" 
    "STANDING"
    "LAYING"
```

一个时间序列有128组数据， 每组9个值。 
```
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_",
```



## 12. 通过图片生成文字解说

* 了解一对多的序列处理问题
* 使用CNN 提取图片的特征，然后输入到LSTM里面进行Decoder。


![image](./assets/im2txt.jpg)
 
 
