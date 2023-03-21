# import tensorflow as tf
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
#conv2d可以改成nn，conv2d

class NonLocalBlockND(layers.Layer):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True,**kwargs):
        super(NonLocalBlockND, self).__init__(**kwargs)
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        self.inter_channels = inter_channels#32
        self.in_channels=in_channels#64
        self.sub_sample=sub_sample
        self.BatchNormalization1=layers.BatchNormalization(axis=-1,name="NonLocalBlockND/bn_inputs")
        self.BatchNormalization2 = layers.BatchNormalization(axis=-1,name="NonLocalBlockND/bn_z")
        self.elu=layers.ELU()
        self.conv1=layers.Conv2D(filters=self.inter_channels,kernel_size=[1,1],strides=[1,1],padding='SAME',name="NonLocalBlockND/conv_g_x")
        self.conv2 = layers.Conv2D(filters=self.inter_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME',name="NonLocalBlockND/conv_thetha_x")
        self.conv3 = layers.Conv2D(filters=self.inter_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME',name="NonLocalBlockND/conv_phi_x")
        self.conv4 = layers.Conv2D(filters=self.in_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME',name="NonLocalBlockND/conv_z")
        self.maxPooling2d = layers.MaxPooling2D(pool_size=[1, 2], strides=[1, 2])
        self.softmax=layers.Softmax()
        self.matmul=layers.Multiply()
        self.permute=layers.Permute((3,1,2))
        self.sita = self.add_weight(shape=(1,),initializer=tf.zeros_initializer(), trainable=True,name="NonLocalBlockND/sita")

    def call(self, inputs, training=False,*args, **kwargs):
        if self.inter_channels is None:
            inter_channels = self.in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        x=self.BatchNormalization1(inputs,training=training)
        x=self.elu(x)
        #g
        g_x=self.conv1(x)
        if(self.sub_sample):
            g_x=self.maxPooling2d(g_x)
        g_x=layers.Reshape((g_x.shape[1]*g_x.shape[2], self.inter_channels))(g_x)#(-1,,)
        #theta
        theta_x=self.conv2(x)
        theta_x=layers.Reshape((theta_x.shape[1]*theta_x.shape[2], self.inter_channels))(theta_x)#(-1,,)
        #phi
        phi_x=self.conv3(x)
        if(self.sub_sample):
            phi_x=self.maxPooling2d(phi_x)
        phi_x = self.permute(phi_x)  # 修改维度，同时改变内容顺序
        phi_x = layers.Reshape((self.inter_channels, phi_x.shape[2] * phi_x.shape[3]))(phi_x)#(-1,,)

        f=tf.matmul(theta_x,phi_x)
        y=self.softmax(f)
        y=tf.matmul(y,g_x)
        y=layers.Reshape((inputs.shape[1], inputs.shape[2], self.inter_channels))(y)
        fre_z=self.BatchNormalization2(y,training=training)
        fre_z=self.elu(fre_z)
        fre_z=self.conv4(fre_z)
        z = fre_z*self.sita + inputs
        return z

# if __name__ == "__main__":
#     img = tf.zeros([100, 1, 101, 64])
#     out = NonLocalBlockND(img, in_channels=64, inter_channels=1)
#     print("out: ")
#     print(out)
#     print(tf.Variable(initial_value=0.0, dtype=tf.float32))
