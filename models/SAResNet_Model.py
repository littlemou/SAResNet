# import tensorflow as tf
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from models.NonLocal_SelfAttention import NonLocalBlockND

class identity_block(layers.Layer):
    def __init__(self, kernel_size, in_filter, out_filters, **kwargs):
        super(identity_block, self).__init__(**kwargs)
        self.BatchNormalization1=layers.BatchNormalization(axis=-1,name="identity_block/bn1")
        self.BatchNormalization2 = layers.BatchNormalization(axis=-1,name="identity_block/bn2")
        self.elu=layers.ELU()
        # self.weight_variable=self.weight_variable([1, kernel_size, in_filter, out_filters])
        self.conv1=layers.Conv2D(in_filter, kernel_size=[1, kernel_size], strides=[1, 1],padding="SAME",name="identity_block/conv1")#strides=[1, 1, stride, 1]
        self.conv2 = layers.Conv2D(in_filter, kernel_size=[1, kernel_size], strides=[1, 1],padding="SAME",name="identity_block/conv2")  # strides=[1, 1, stride, 1]
        self.add=layers.Add()

    def call(self, inputs, training=False, *args, **kwargs):
        #第一次
        x=self.BatchNormalization1(inputs,training=training)
        x=self.elu(x)
        x=self.conv1(x)
        #第二次
        x = self.BatchNormalization2(x,training=training)
        x = self.elu(x)
        x = self.conv2(x)
        #首尾相加
        x = self.add([x, inputs])  # 结果和原始相加
        return x

class convolutional_block(layers.Layer):
    def __init__(self, kernel_size, in_filter, out_filters,stride=2,**kwargs):
        super(convolutional_block, self).__init__(**kwargs)
        self.BatchNormalization1 = layers.BatchNormalization(axis=3,name="convolutional_block/bn1")
        self.BatchNormalization2 = layers.BatchNormalization(axis=3, name="convolutional_block/bn2")
        self.elu = layers.ELU()
        # self.weight_variable=self.weight_variable([1, kernel_size, in_filter, out_filters])
        # print(out_filters%1)
        self.conv1 = layers.Conv2D(filters=in_filter, kernel_size=[1, kernel_size], strides=[1, stride],padding="SAME",name="convolutional_block/conv1")#strides=[1, 1, stride, 1]
        self.conv2 = layers.Conv2D(filters=in_filter, kernel_size=[1, kernel_size], strides=[1, 1],padding="SAME",name="convolutional_block/conv2")#strides=[1, 1, 1, 1]
        self.conv3=layers.Conv2D(filters=in_filter, kernel_size=[1,1], strides=[1, stride],padding="SAME",name="convolutional_block/conv3")#strides=[1, 1, stride, 1
        self.add = layers.Add()

    def call(self, inputs,training=False, *args, **kwargs):
        #第一次
        x=self.BatchNormalization1(inputs,training=training)
        x=self.elu(x)
        x=self.conv1(x)
        #第二次
        x = self.BatchNormalization2(x,training=training)
        x = self.elu(x)
        x = self.conv2(x)

        #变换初始x
        x_shortcut=self.conv3(inputs)
        x=self.add([x,x_shortcut])
        return x


class SAResnetModel(tf.keras.Model):
    def __init__(self, kernal_channel, fc_num,**kwargs):
        super(SAResnetModel, self).__init__(**kwargs)
        self.kernal_channel=kernal_channel
        self.fc_num=fc_num
        self.conv1=layers.Conv2D(filters=self.kernal_channel,kernel_size=[1,7], strides=[1, 1],padding="SAME",name="SAResnetModel/conv1")
        self.elu=layers.ELU()
        self.maxPooling2d=layers.MaxPooling2D(pool_size=[1, 3], strides=[1, 2])
        self.SAResBlock1 = make_layer(conv_block_num=1,iden_block_num=2, kernal_channel=self.kernal_channel,have_Attention=True,stride=1, name="SAResnetModel/SAResBlock1")
        self.SAResBlock2 = make_layer(conv_block_num=1, iden_block_num=3, kernal_channel=self.kernal_channel, have_Attention=True,stride=2,name="SAResnetModel/SAResBlock2")
        self.SAResBlock3 = make_layer(conv_block_num=1, iden_block_num=5, kernal_channel=self.kernal_channel, have_Attention=False,stride=2,name="SAResnetModel/SAResBlock3")
        self.SAResBlock4 = make_layer(conv_block_num=1, iden_block_num=2, kernal_channel=self.kernal_channel,have_Attention=False, stride=2, name="SAResnetModel/SAResBlock4")
        # self.NonLocalBlock=NonLocalBlockND(in_channels=self.kernal_channel, sub_sample=False)
        # self.ConvolutionalBlock=convolutional_block(3, self.kernal_channel, [self.kernal_channel, self.kernal_channel], stride=1)
        # self.IdentityBlock=identity_block(3, self.kernal_channel, [self.kernal_channel, self.kernal_channel])
        self.averagePooling=layers.AveragePooling2D(pool_size=[1, 7], strides=[1, 1])
        self.flatten=layers.Flatten()
        self.dropout=layers.Dropout(rate=0.7)
        self.dense1=layers.Dense(units=self.fc_num, activation=None,name="SAResnetModel/dense1")
        self.dense_out = layers.Dense(units=2, activation=None,name="SAResnetModel/dense_out")
        self.softmax=layers.Softmax()

    # def parse(self,logits):
    #     for i in range(logits.shape[0]):
    #         if(logits[i][0]>lo)

    def call(self, inputs, training=None, mask=None):
        #对输入进行加深通道
        # x=layers.Reshape(([1, self.weight, self.channels]))(inputs)
        # print(x.shape)
        # self.input_x = tf.reshape(self.input_x, [-1, 1, self.weight, self.channels])
        # print(inputs.shape)
        x=self.conv1(inputs)
        # print(x.shape)
        x=self.elu(x)
        x=self.maxPooling2d(x)
        # print(x.shape)

        x = self.SAResBlock1(x, training=training)
        # print("1")
        # print(x.shape)
        x = self.SAResBlock2(x, training=training)
        # print("2")
        # print(x.shape)
        x = self.SAResBlock3(x, training=training)
        # print("3")
        # print(x.shape)
        x = self.SAResBlock4(x, training=training)
        # print("4")
        # print(x.shape)
        #最后输出层
        x=self.averagePooling(x)
        flatten=self.flatten(x)
        x=self.dropout(flatten,training=training)
        x=self.dense1(x)
        x = self.dropout(x,training=training)
        logits = self.dense_out(x)
        logits=self.softmax(logits)
        # print(logits)
        # model_predict = tf.argmax(logits, axis=1)
        # print(model_predict)
        # model_predict = tf.cast(model_predict, dtype=tf.int32)
        return logits


# 将相同的残差模块（比如Bottlenect）连接成一个大的残差块，对于原论文表1中的conv2_x、conv3_x、conv4_x、conv5_x
def make_layer(iden_block_num,conv_block_num, kernal_channel,stride, have_Attention,name):
    # 实现表1中的一个大的残差块
    layers_list = []
    if(have_Attention):
        layers_list.append(NonLocalBlockND(in_channels=kernal_channel, sub_sample=False,name="unit_nonLocal"))#每个块先加一个non-local增强数据焦点
    for i in range(1,conv_block_num+1):
        layers_list.append(convolutional_block(3, kernal_channel, [kernal_channel, kernal_channel], stride=stride,name="unit_conv_block"+str(i)))
    for i in range(conv_block_num+1, iden_block_num+conv_block_num+1):
        # print("unit_iden_block"+str(i))
        layers_list.append(identity_block(3, kernal_channel, [kernal_channel, kernal_channel],name="unit_iden_block"+str(i)))

    return Sequential(layers_list, name=name)