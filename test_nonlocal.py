import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Reshape, Multiply, Lambda


class NonLocalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(NonLocalSelfAttention, self).__init__(**kwargs)
        self.channels = channels

        # 线性变换，将输入的特征图映射到query、key、value
        self.query_conv = Conv2D(filters=channels // 8, kernel_size=1, activation=None, use_bias=False)
        self.key_conv = Conv2D(filters=channels // 8, kernel_size=1, activation=None, use_bias=False)
        self.value_conv = Conv2D(filters=channels, kernel_size=1, activation=None, use_bias=False)

        # 残差连接相关的层
        self.gamma = self.add_weight(shape=[], initializer=tf.zeros_initializer(), trainable=True, name="gamma")
        self.beta = self.add_weight(shape=[], initializer=tf.ones_initializer(), trainable=True, name="beta")

    def build(self, input_shape):
        super(NonLocalSelfAttention, self).build(input_shape)

    def call(self, input_tensor):
        # 计算query、key、value
        query = self.query_conv(input_tensor)
        key = self.key_conv(input_tensor)
        value = self.value_conv(input_tensor)

        # 获取特征图的宽和高
        h, w = input_tensor.shape[1], input_tensor.shape[2]

        # 将query、key进行reshape，以便计算相似度
        query = Reshape((-1, h * w))(query)
        key = Reshape((-1, h * w))(key)

        # 计算query和key之间的相似度，并归一化
        sim_map = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))(inputs=[query, key])
        sim_map = Lambda(lambda x: x / tf.math.sqrt(tf.cast(self.channels, dtype=tf.float32)))(sim_map)
        sim_map = tf.nn.softmax(sim_map)

        # 计算每个位置的权重，即value的加权和的权重
        attention = Multiply()([sim_map, value])
        attention = Reshape((h, w, -1))(attention)
        attention = tf.reduce_sum(attention, axis=[1, 2])
        attention = Dense(units=self.channels)(attention)

        # 进行残差连接
        output = tf.multiply(attention, self.gamma) + input_tensor * self.beta

        return output
