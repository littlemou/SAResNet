import os
import tensorflow as tf
from data_loader.load_data import load_data,parse
from models.SAResNet_Model import SAResnetModel
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import CosineDecay
import numpy as np
import xlwt
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
curPath=os.path.split(curPath)[0]
# sys.path.append(os.path.split(curPath)[0])
# print(os.path.split(curPath)[0])
class SAResNetTrainer():
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
        model = SAResnetModel(kernal_channel=kernal_channel, fc_num=fc_num)
        model.load_weights(filepath=ckpt_path)
        model.trainable = True
        model.build((None, 1, 101, 4))  # when using subclass model
        return model

    def train_save(self,epochs,ckpt_save_path):
        best_test_loss = float('inf')
        # xlsSave_path=os.path.join(curPath,"log")
        xlsSave_path = os.path.join(ckpt_save_path,"SAResNet"+"_batchSize="+str(self.batch_size)+"_learnRate="+str(self.learning_rate)+".xls")
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
                print(os.path.join(ckpt_save_path , "SAResNet.ckpt"))
                self.model.save_weights(os.path.join(ckpt_save_path, "SAResNet.ckpt"), save_format="tf")
                print("save the best model,vlaid loss=" + str(best_test_loss))
            #记录
            self.out_sheet.write(epoch, 0, epoch)
            self.out_sheet.write(epoch, 1, float(self.test_loss.result()))
            self.out_sheet.write(epoch, 2, float(self.test_accuracy.result()))
            # self.out_book.save("SAResNet"+"_batchSize="+str(self.batch_size)+"_learnRate="+str(self.learning_rate)+".xls")
            self.out_book.save(xlsSave_path)
            # if(num>=1):#动态学习率
            self.learning_rate=self.learning_rate*0.5
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            if(num>=5 ):#and float(self.test_accuracy.result())>=0.85
              num=0
              break

if __name__=='__main__':
    data_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\data_pretrain\150_global\trans_data"
    (train, train_label), (test, test_label), (valid, valid_label) = load_data(data_path)
    print("——————————加载数据结束")
    kernal_channel=64
    fc_num=32
    batch_size=64
    epoch=10
    model=SAResnetModel(kernal_channel=kernal_channel,fc_num=fc_num)
    ckpt_save_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\save_weights"
    ckpt_name="SAResNet.ckpt"
    trainer=SAResNetTrainer(model=model,data_path=data_path,batch_size=batch_size)
    trainer.train_save(epochs=epoch,ckpt_save_path=ckpt_save_path)

    ckpt_path=os.path.join(ckpt_save_path,ckpt_name)

# batch_size=100
# project_path=os.path.dirname(os.path.dirname(__file__))
# data_path=os.path.join(project_path,"DataSet","tfrecords_pretrain","global_tfrecords")


# (train,train_label),(test,test_label),(valid,valid_label)=load_data(data_path)
# print("——————————加载数据结束")
# train_dataset = tf.data.Dataset.from_tensor_slices((train,train_label))
# train_dataset = train_dataset.shuffle(50000).map(parse,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(128)
#
# test_dataset = tf.data.Dataset.from_tensor_slices((test,test_label))
# test_dataset = test_dataset.map(parse,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(128)

# sample = next(iter(train_dataset))
# print('sample:', sample[0].shape, sample[1].shape,
#       tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

#################################################################
#
# class ExampleTrainer():
#     def __init__(self, model, data_path):
#         self.model=model
#         self.data_path=data_path
#         (train, train_label), (test, test_label), (valid, valid_label) = load_data(data_path)
#         print("——————————加载数据结束")
#         train_dataset = tf.data.Dataset.from_tensor_slices((train, train_label))
#         self.train_dataset = train_dataset.shuffle(50000).map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)
#
#         test_dataset = tf.data.Dataset.from_tensor_slices((test, test_label))
#         self.test_dataset = test_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=64)
#         # self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
#         # self.loss_softmax=tf.nn.softmax_cross_entropy_with_logits()
#         self.optimizer = tf.keras.optimizers.Adam()
#
#     def train_step(self,inputs, labels):
#         with tf.GradientTape() as tape:
#             predictions = self.model(inputs,training=True)
#             # print(predictions)
#             loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=labels)
#             loss=tf.reduce_mean(loss)
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#         return loss
#
#     def test_step(self,inputs, labels):
#         predictions = self.model(inputs,training=False)
#         # prediction = tf.argmax(prediction, axis=1)
#         # prediction = tf.cast(prediction, dtype=tf.int32)
#         # correct_prediction = tf.cast(tf.equal(prediction, labels), dtype=tf.int32)
#         # correct = tf.reduce_sum(correct_prediction)
#         # total_num += x.shape[0]
#         # total_correct += int(correct)
#
#         # acc = total_correct / total_num
#         # print('第', epoch + 1, '个epoch的测试精确度：acc =', acc)
#         loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=labels)
#         loss = tf.reduce_mean(loss)
#         return loss
#
#     def trainModel(self):
#         for epoch in range(10):
#             print('Epoch:', epoch + 1)
#             for step, (inputs, labels) in enumerate(self.train_dataset):
#                 loss = self.train_step(inputs, labels)
#                 if step % 100 == 0:
#                     print('Step:', step, 'Loss:', loss.numpy())
#
#             for test_inputs, test_labels in self.test_dataset:
#                 test_loss = self.test_step(test_inputs, test_labels)
#
#             print('Test Loss:', test_loss.numpy())
#
#             total_num = 0
#             total_correct = 0
#             for x, y in self.test_dataset:
#                 logits = self.model(x, training=False)
#
#                 test_predictions = tf.argmax(logits, axis=1)#onehot转换为单个标签
#                 test_predictions = tf.cast(test_predictions, dtype=tf.int32)
#
#                 y = tf.argmax(y, axis=1)#onehot转换为单个标签
#                 y=tf.cast(y,dtype=tf.int32)
#                 correct = tf.cast(tf.equal(test_predictions, y), dtype=tf.int32)
#                 correct = tf.reduce_sum(correct)
#                 total_num += x.shape[0]
#                 total_correct += int(correct)
#             acc = total_correct / total_num
#             print('第', epoch + 1, '个epoch的测试精确度：acc =', acc)
#
    # def trainModel_2(self):


