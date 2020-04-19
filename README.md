# 学习RNN神经网络


循环神经网络可以用来解决各种序列问题， 本项目给出一些常见的循环神经网络的使用场景和代码实现。 

![image](https://dikers-data.s3.cn-northwest-1.amazonaws.com.cn/images/rnn.png)




## 1. 用numpy实现简单的RNN

实现一个简单的RNN， 预测字符串序列, 理解一些常用的矩阵运算， 以及前向、反向传播


以下是RNN的计算公式
![image](https://pic1.zhimg.com/80/v2-4058db6817f202ddc3fc41cb3683a744_1440w.png)




## 2. 对比LSTM 和RNN

* 理解序列问题的数据格式
* 对比LSTM 处理长序列问题的能力

```
 s=0.7581    m=1
 s=0.7010    m=0
 s=0.1807    m=0
 s=0.8387    m=0
 s=0.6760    m=0
 s=0.3419    m=0
 ......
 s=0.0420    m=0
 s=0.8913    m=0
 s=0.3688    m=1
 s=0.8188    m=0
```

 s列和m列想乘 再求和 等于y = 1.1269， 其中有两个m是1，剩余都为0
 
 这个序列的长度会达到比较长， 如果用普通RNN很难发现规律，体验LSTM的威力



## 3. 使用Tensorflow 实现RNN 预测Sin函数图像

* 理解序列问题数据处理方式， X 和y的数据格式。 
* 熟悉Tensorflow 构建网络模型的常规流程。 


## 4. 使用LSTM 对mnist 手写数字集进行分类

将每行数据看做时序序列对图像进行分类。 

![image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1585420330395&di=fe0f331f114848935b109a657958fd6c&imgtype=0&src=http%3A%2F%2Finews.gtimg.com%2Fnewsapp_bt%2F0%2F6847845171%2F1000)




## 5. 实现自己的LSTM 

通过实现LSTM， 理解LSTM cell的内部结构， 加深理解。 

![image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1586192304337&di=ffda304e49c896afcdd4347fa8b6ad7c&imgtype=0&src=http%3A%2F%2Ffile.elecfans.com%2Fweb1%2FM00%2FA4%2FE3%2Fo4YBAF1o35KADzDyAAC0_ErNSG8845.png)
![image](https://img2018.cnblogs.com/blog/1575054/201904/1575054-20190412231515774-190310739.png)



## 6. 通过Encoder Decoder实现问答系统

序列到序列的模型， 核心就是研究如何处理序列的问题。 

```
Q: 24+654
A: 678
```
程序读取问题字符串，通过和答案的对比， 发现序列里面的规律。  



## 7. 使用PyTorch实现字符级别的英语文章预测

以下是预测出的文本
```
The moss of the convincing it had been drawing up the people that there was nothing without 
this way or a single wife as he did not hear him or that he was not seeing that she would 
be a court of the sound of some sound of the position, and to spartly she could see her 
and a sundroup times there was nothing this father and as she stoop serious in the sound, 
was a steps of the master, a few sistersily play of his husband. The crowd had no carreated
herself, and truets, and shaking up, the pases, and the moment that he was not at the marshal,
and the starling the secret were stopping to be
```


## 8. 中文文章的分类

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



## 9. 英语到西班牙的翻译功能

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



## 10. 自动作曲

* 熟悉声音文件的处理
* 熟悉声音序列问题的处理方式



## 11. 自动完成阅读理解

通过记忆网络实现问答系统， 记忆网络将信息保存到网络外部， 类比LSTM的长期记忆， 范围和尺度还要更广。 


```
Daniel moved to the garden . Mary went back to the bathroom .

Q: Where is Daniel ?

A: garden
```


```
John went to the bedroom . Daniel journeyed to the kitchen . Daniel journeyed to the hallway . 
Mary travelled to the bathroom . John travelled to the garden . Daniel journeyed to the office .
Mary moved to the bedroom . Sandra went back to the hallway .

Q: Where is Sandra ?

A: hallway

```


## 12. 通过手机传感器预测人类运动

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




## 13. 通过图片生成文字解说

* 了解一对多的序列处理问题
* 使用CNN 提取图片的特征，然后输入到LSTM里面进行Decoder。


![image](./assets/im2txt.jpg)
 
 


## 14. 生成周杰伦的歌词

* 使用Mxnet实现LSTM
* 熟悉LSTM 的公式

以下是在小数据集上，迭代了500次以后生成的歌词
>  喜欢 问候我 谁是神枪手 巫师 他念念 有词的 对酋长下诅咒 还我骷髅头 这故事 告诉我 印地安的传说 
>  
>  不分开 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活



## 15. 图像生成HTML代码

* 了解CNN和RNN的混合使用
* Keras 使用多输入模型

![image](https://camo.githubusercontent.com/7aee7deacf38b8f9a2a230da4efbd96a96840b83/68747470733a2f2f692e696d6775722e636f6d2f4c446d6f4c4c562e706e67)


## 16. 命名实体识别

使用《射雕三部曲》中主要人物、武功、门派之间的关系来构建知识图谱。
![image](./assets/016.jpg)


* 了解命名实体识别

样本数据如下：
```
张无忌，金庸武侠小说《倚天屠龙记》人物角色，中土明教第三十四代教主。武当七侠之一张翠山与天鹰教紫微堂主殷素素之子，明教四大护教法王之一金毛狮王谢逊义子。
张翠山，《倚天屠龙记》第一卷的男主角，在武当七侠之中排行第五，人称张五侠。与天鹰教殷素素结为夫妇，生下张无忌，后流落到北极冰海上的冰火岛，与谢逊相识并结为兄弟。
殷素素，金庸武侠小说《倚天屠龙记》第一卷的女主人公。天鹰教紫薇堂堂主，容貌娇艳无伦，智计百出，亦正亦邪。与武当五侠张翠山同赴王盘山，结果被金毛狮王谢逊强行带走，三人辗转抵达冰火岛。殷素素与张翠山在岛上结为夫妇，并诞下一子张无忌。谢逊，是金庸武侠小说《倚天屠龙记》中的人物，字退思，在明教四大护教法王中排行第三，因其满头金发，故绰号“金毛狮王”。
                
 {'name': ['张无忌', '张翠山', '殷素素', '谢逊'],
  'book': ['倚天屠龙记'],
  'org': ['明教', '武当', '天鹰教']})
```

* 了解实体关系抽取

样本数据如下：
```
([['杨康', '杨铁心', '子女', '杨康，杨铁心与包惜弱之子，金国六王爷完颜洪烈的养子。'],
  ['杨康', '杨铁心', '子女', '丘处机与杨铁心、郭啸天结识后，以勿忘“靖康之耻”替杨铁心的儿子杨康取名。'],
  ['杨铁心', '包惜弱', '配偶', '金国六王爷完颜洪烈因为贪图杨铁心的妻子包惜弱的美色，杀害了郭靖的父亲郭啸天。'],
  ['杨铁心', '包惜弱', '配偶', '杨康，杨铁心与包惜弱之子，金国六王爷完颜洪烈的养子。'],
  ['张翠山', '殷素素', '配偶', '张无忌,武当七侠之一张翠山与天鹰教紫微堂主殷素素之子。'],
  ['小龙女', '杨过', '师傅', '小龙女是杨过的师父，与杨过互生情愫，但因师生恋不容于世。'],
  ['黄药师', '黄蓉', '父', '黄药师，黄蓉之父，对其妻冯氏（小字阿衡）一往情深。'],
  ['郭啸天', '郭靖', '父', '郭靖之父郭啸天和其义弟杨铁心因被段天德陷害，死于临安牛家村。']],
 {'子女': 0, '配偶': 1, '师傅': 2, '父': 3})
```

