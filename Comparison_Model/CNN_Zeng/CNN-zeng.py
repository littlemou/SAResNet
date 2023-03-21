# import tensorflow as tf
import os
import tensorflow as tf
import xlrd
import xlwt
from tensorflow import keras
from tensorflow.keras import layers,Sequential

class CNN_zeng(tf.keras.Model):
    def __init__(self, kernal_channel, fc_num,**kwargs):
        super(CNN_zeng, self).__init__(**kwargs)
        self.kernal_channel=kernal_channel
        self.fc_num=fc_num
        self.conv1=layers.Conv2D(filters=self.kernal_channel,kernel_size=7, strides=[1, 1],padding="SAME",name="CNN_zeng/conv1")
        self.elu=layers.ELU()
        self.maxPooling2d=layers.MaxPooling2D(pool_size=[1, 3], strides=[1, 2])
        self.CNN_zengBlock1 = make_layer(kernal_channel=self.kernal_channel, name="CNN_zeng/CNN_zengBlock1")
        self.CNN_zengBlock2 = make_layer( kernal_channel=self.kernal_channel, name="CNN_zeng/CNN_zengBlock2")
        self.CNN_zengBlock3 = make_layer( kernal_channel=self.kernal_channel, name="CNN_zeng/CNN_zengBlock3")
        self.averagePooling=layers.AveragePooling2D(pool_size=[ 1, 7], strides=[1, 1])
        self.flatten=layers.Flatten()
        self.dropout=layers.Dropout(rate=0.5)
        self.dense1=layers.Dense(units=self.fc_num, activation=None,name="CNN_zeng/dense1")
        self.dense_out = layers.Dense(units=2, activation=None,name="CNN_zeng/dense_out")
        self.softmax=layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        #对输入进行加深通道
        # x=self.conv1(inputs)
        # x=self.elu(x)
        # x=self.maxPooling2d(x)

        x = self.CNN_zengBlock1(inputs, training=training)
        x = self.CNN_zengBlock2(x, training=training)
        x = self.CNN_zengBlock3(x, training=training)
        #最后输出层
        # x=self.averagePooling(x)
        flatten=self.flatten(x)
        # x=self.dropout(flatten,training=training)
        # x=self.dense1(x)
        x = self.dropout(flatten,training=training)
        logits = self.dense_out(x)
        logits=self.softmax(logits)
        return logits


# 将相同的残差模块（比如Bottlenect）连接成一个大的残差块，对于原论文表1中的conv2_x、conv3_x、conv4_x、conv5_x
def make_layer( kernal_channel, name):
    # 实现表1中的一个大的残差块
    layers_list = []
    layers_list.append(
        layers.Conv2D(filters=kernal_channel,kernel_size=7, strides=[1, 1],padding="SAME",name="unit_conv_block1"))
    layers_list.append(
        layers.BatchNormalization(axis=-1, name="unit_conv_block2"))
    layers_list.append(
        layers.MaxPooling2D(pool_size=[1, 3], strides=[1, 2]))
    return Sequential(layers_list, name=name)


import numpy as np
from load_data import *


class CNN_zengTrainer():
    def __init__(self, model, data_path,batch_size=128,learning_rate=0.0001):
        self.original_model=model
        self.model=model
        self.data_path=data_path

        (self.train, self.train_label), (self.test, self.test_label), (self.valid, self.valid_label) = load_data(data_path)
        self.train_num= len(self.train)
        self.test_num = len(self.test)
        self.valid_num = len(self.valid)
        self.batch_size=batch_size
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train, self.train_label))
        self.train_dataset = train_dataset.shuffle(50000).map(parse,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=self.batch_size)
        # test_dataset = tf.data.Dataset.from_tensor_slices((self.test, self.test_label))
        # self.test_dataset = test_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((self.valid, self.valid_label))
        self.valid_dataset = valid_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=self.batch_size)
        print("——————————加载数据结束")
        print("——————————训练数据数量：" + str(self.train_num))
        print("——————————验证数据数量：" + str(self.valid_num))
        print("——————————batch_size：" + str(self.batch_size))
        print("——————————总批次数：" + str(self.train_num // self.batch_size))

        # 1.目标损失函数：交叉熵
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # 2.优化器：Adam
        self.learning_rate=learning_rate
        # lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # 3.评价标准：loss和accuracy
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        # tf.keras.metrics.CategoricalCrossentropy
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        self.out_book = xlwt.Workbook()
        self.out_sheet = self.out_book.add_sheet('Sheet1')
        self.out_sheet.write(0,0,"epochs")
        self.out_sheet.write(0, 1, "loss")
        self.out_sheet.write(0, 2, "accuracy")
        #test

    def train_step(self,inputs, labels):
        with tf.GradientTape() as tape:  # 建立梯度环境
            output = self.model(inputs, training=True)  # 前向计算
            loss = self.loss_object(labels, output)  # 计算目标损失
        gradients = tape.gradient(loss, self.model.trainable_variables)  # 自动求梯度
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  # 反向传播，更新参数
        self.train_loss(loss)  # 计算训练集的平均损失值loss
        self.train_accuracy(labels, output)  # 计算训练集上的准确性accuracy

    def test_step(self,inputs, labels):
        output = self.model(inputs, training=False)  # 前向计算
        t_loss = self.loss_object(labels, output)  # 求每一次的目标损失值
        self.test_loss(t_loss)  # 求平均损失值
        self.test_accuracy(labels, output)

    def display_train_progress(self,epoch,step,total_num):
        # 打印训练过程
        rate = (step + 1) / (total_num // self.batch_size)  # 一个epoch中steps的训练完成度
        if(np.mod(int(rate*100),10)!=0):
          return
        a = "*" * int(rate * 50)  # 已完成进度条用*表示
        b = "." * int((1 - rate) * 50)  # 未完成进度条用.表示
        acc = self.train_accuracy.result().numpy()
        print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
        print('\n')

    def display_test_result(self, epoch):
        #  每训练完一个epoch后，打印显示信息
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              self.train_loss.result(),
                              self.train_accuracy.result() * 100,
                              self.test_loss.result(),
                              self.test_accuracy.result() * 100))

    def loadModel(self,ckpt_path, kernal_channel=64, fc_num=32):
        model = CNN_zeng(kernal_channel=kernal_channel, fc_num=fc_num)
        model.load_weights(filepath=ckpt_path)
        model.trainable = True
        model.build((None, 1, 101, 4))  # when using subclass model
        return model

    def train_save(self,epochs,ckpt_save_path):
        best_test_loss = float('inf')
        # xlsSave_path=os.path.join(curPath,"log")
        xlsSave_path = os.path.join(ckpt_save_path,"CNN_zeng.xls")
        # print(xlsSave_path)
        num=0
        for epoch in range(1, epochs + 1):
            # print(self.optimizer.)
            # 重置清零每一轮的loss值和accuracy。因此后面打印出来的是每一个epoch中的loss、accuracy平均值，而不是历史平均值。
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            # 训练
            for step, (inputs, labels) in enumerate(self.train_dataset):
                self.train_step(inputs, labels)
                self.display_train_progress(epoch, step, self.train_num)
            # 验证集测试过程
            for step, (inputs, labels) in enumerate(self.valid_dataset):
                self.test_step(inputs, labels)
            self.display_test_result(epoch)
            num+=1
            if self.test_loss.result() < best_test_loss:
                num=0
                best_test_loss = self.test_loss.result()
                # 保存模型参数
                print(os.path.join(ckpt_save_path,"CNN_zeng.ckpt"))
                self.model.save_weights(os.path.join(ckpt_save_path, "CNN_zeng.ckpt"), save_format="tf")
                print("save the best model,vlaid loss=" + str(best_test_loss))
            #记录
            self.out_sheet.write(epoch, 0, epoch)
            self.out_sheet.write(epoch, 1, float(self.test_loss.result()))
            self.out_sheet.write(epoch, 2, float(self.test_accuracy.result()))
            # self.out_book.save("SAResNet"+"_batchSize="+str(self.batch_size)+"_learnRate="+str(self.learning_rate)+".xls")
            self.out_book.save(xlsSave_path)

            if(num>=5 and float(self.test_accuracy.result())>=0.75):
              num=0
              break

from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class Evaluater():
    def __init__(self, model, data_path,batch_size=5000):
        self.model=model
        (self.test, self.test_label)= load_data_evaluate(data_path)
        self.test_num = len(self.test)
        self.batch_size = batch_size
        test_dataset = tf.data.Dataset.from_tensor_slices((self.test, self.test_label))
        self.test_dataset = test_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=self.batch_size)#一次性全部

    def test_step(self, inputs, labels):
        output = self.model(inputs, training=False)  # 前向计算
        labels = tf.argmax(labels, axis=1)
        labels = tf.cast(labels, dtype=tf.int32)
        output=tf.argmax(output,axis=1)
        output = tf.cast(output, dtype=tf.int32)

        auc=roc_auc_score(labels,output)
        acc = accuracy_score(labels, output)
        precision = precision_score(labels, output)
        recall = recall_score(labels, output)
        f1 = f1_score(labels, output)

        return acc,precision,recall,f1,auc
    def testModel(self):
        Accuary=0
        Precision=0
        Recall=0
        F1=0
        AUC=0
        num=0
        for step, (inputs, labels) in enumerate(self.test_dataset):
            # print("————————————预测步骤")
            num+=1
            acc, precision, recall, f1,auc=self.test_step(inputs, labels)
            AUC += auc
            Accuary+=acc
            Precision+=precision
            Recall+=recall
            F1+=f1
        Accuary = Accuary/num
        Precision = Precision/num
        Recall = Recall/num
        F1 = F1/num
        AUC = AUC/num
        return Accuary, Precision, Recall, F1 ,AUC

def loadModel(ckpt_path, kernal_channel=64, fc_num=32):
    model = CNN_zeng(kernal_channel=kernal_channel, fc_num=fc_num)
    model.load_weights(filepath=ckpt_path)
    model.trainable = True
    model.build((None, 1, 101, 4))  # when using subclass model
    return model

def evaluate():
    print("————————————预测global模型ing")
    # return
    logpath=os.path.join(os.getcwd(),"log")
    subCkpt_path=os.path.join(os.getcwd(), "model_checkpoint")
    subData_path=r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\data_sub"
    subAUC_list = []
    ave_accuary = 0
    ave_precision = 0
    ave_recall = 0
    ave_f1 = 0
    ave_auc = 0
    num = 0

    out_book = xlwt.Workbook()
    out_sheet = out_book.add_sheet('Sheet1')
    out_sheet.write(0, 0, "TransferModelName")
    out_sheet.write(0, 1, "Accuary")
    out_sheet.write(0, 2, "Precision")
    out_sheet.write(0, 3, "Recall")
    out_sheet.write(0, 4, "F1")
    out_sheet.write(0, 5, "AUC")
    out_book.save(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))
    for file in os.listdir(subData_path):
        print(str(num) + "]当前file：" + file)
        now_ckpt = os.path.join(subCkpt_path, file, "CNN_zeng.ckpt")
        now_model = loadModel(now_ckpt, kernal_channel=64, fc_num=32)
        now_file = os.path.join(subData_path, file)
        sub_evaluater = Evaluater(now_model, now_file)
        Accuary, Precision, Recall, F1, AUC = sub_evaluater.testModel()
        ave_accuary += Accuary
        ave_precision += Precision
        ave_recall += Recall
        ave_f1 += F1
        ave_auc += AUC
        num += 1
        subAUC_list.append(AUC)

        out_sheet.write(num, 0, file)
        out_sheet.write(num, 1, Accuary)
        out_sheet.write(num, 2, Precision)
        out_sheet.write(num, 3, Recall)
        out_sheet.write(num, 4, F1)
        out_sheet.write(num, 5, AUC)
        out_book.save(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))
    ave_accuary = ave_accuary / num
    ave_precision = ave_precision / num
    ave_recall = ave_recall / num
    ave_f1 = ave_f1 / num
    ave_auc = ave_auc / num

    # 写入平均
    num += 1
    out_sheet.write(num, 0, "平均值")
    out_sheet.write(num, 1, ave_accuary)
    out_sheet.write(num, 2, ave_precision)
    out_sheet.write(num, 3, ave_recall)
    out_sheet.write(num, 4, ave_f1)
    out_sheet.write(num, 5, ave_auc)
    out_book.save(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))

    # return ave_accuary, ave_precision, ave_recall, ave_f1, ave_auc

if __name__=='__main__':
    trainData_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\data_sub"
    ckpt_save_path = os.path.join(os.getcwd(), "model_checkpoint")
    num = 0
    for file in os.listdir(trainData_path):
        num += 1
        if(num<106):
            continue
        print(str(num) + "]当前file：" + file)
        now_trainData_Path = os.path.join(trainData_path, file)
        now_ckpt_save_path = os.path.join(ckpt_save_path, file)
        if not os.path.exists(now_ckpt_save_path):
            os.makedirs(now_ckpt_save_path)
        # xls_path = os.path.join(now_ckpt_save_path, "CNN_zeng.xls")
        model =CNN_zeng(kernal_channel=64, fc_num=32)
        trainer = CNN_zengTrainer(model=model, data_path=now_trainData_Path, batch_size=128, learning_rate=0.00001)
        trainer.train_save(epochs=30, ckpt_save_path=now_ckpt_save_path)

    evaluate()