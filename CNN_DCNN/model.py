import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Model(object):
    def __init__(self, batch_size=8, learning_rate=0.0005, num_labels=56):#24232;18908
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels

    def inference(self, images, keep_prob):#The input shape is 208*340*2
        
        with tf.variable_scope('conv0') as scope:
            kernel0 = self._create_weights([4, 8, 2, 16])
            conv0 = self._create_conv2d_V2(images, kernel0)
            bias0 = self._create_bias([16])
            preactivation = tf.nn.bias_add(conv0, bias0)
            conv0 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(conv0)#103*167*16          
        with tf.variable_scope('conv1') as scope:
            kernel1 = self._create_weights([5, 7, 16, 32])
            conv1 = self._create_conv2d_V2(conv0, kernel1)
            bias1 = self._create_bias([32])
            preactivation = tf.nn.bias_add(conv1, bias1)
            conv1 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(conv1)#50*81*32           

        with tf.variable_scope('conv2') as scope:
            kernel2 = self._create_weights([4, 7, 32, 64])
            conv2 = self._create_conv2d_V2(conv1, kernel2)
            bias2 = self._create_bias([64])
            preactivation = tf.nn.bias_add(conv2, bias2)
            conv2 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(conv2)#24*38*64

        with tf.variable_scope('conv3') as scope:
            kernel3 = self._create_weights([4, 8, 64, 128])
            conv3 = self._create_conv2d_V2(conv2, kernel3)
            bias3 = self._create_bias([128])
            preactivation = tf.nn.bias_add(conv3, bias3)
            conv3 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(conv3)#11*16*128
        
        with tf.variable_scope('conv4') as scope:
            kernel4 = self._create_weights([3, 4, 128, 256])
            conv4 = self._create_conv2d_V2(conv3, kernel4)
            bias4 = self._create_bias([256])
            preactivation = tf.nn.bias_add(conv4, bias4)
            conv4 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(conv4)#5*7*256
            
        with tf.variable_scope('conv5') as scope:
            kernel5 = self._create_weights([2, 3, 256, 512])
            conv5 = self._create_conv2d_V(conv4, kernel5)
            bias5 = self._create_bias([512])
            preactivation = tf.nn.bias_add(conv5, bias5)
            conv5 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(conv5)#4*5*512
        #deconv
        with tf.variable_scope('deconv0') as scope:
            dekernel0 = self._create_weights([2, 3, 256, 512])
            output_shape0 = tf.stack([tf.shape(images)[0],5,7,256])
            deconv0 = self._create_deconv1(conv5, dekernel0, output_shape0)
            debias0 = self._create_bias([256])
            preactivation = tf.nn.bias_add(deconv0, debias0)#5*7*256
            deconv0 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(deconv0)
            
        with tf.variable_scope('deconv1') as scope:
            dekernel1 = self._create_weights([3, 4, 128, 256])
            output_shape1 = tf.stack([tf.shape(images)[0],11,16,128])
            deconv1 = self._create_deconv(deconv0, dekernel1, output_shape1)
            debias1 = self._create_bias([128])
            preactivation = tf.nn.bias_add(deconv1, debias1)#11*16*128
            deconv1 = tf.nn.elu(preactivation, name=scope.name)
            #self._activation_summary(deconv1)
            
        with tf.variable_scope('deconv2') as scope:
            dekernel2 = self._create_weights([4, 8, 64, 128])
            output_shape2 = tf.stack([tf.shape(images)[0],24,38,64])
            deconv2 = self._create_deconv(deconv1, dekernel2, output_shape2)
            debias2 = self._create_bias([64])
            preactivation2 = tf.nn.bias_add(deconv2, debias2)#24*38*64
            deconv2 = tf.nn.elu(preactivation2, name=scope.name)
            #self._activation_summary(deconv2)

        with tf.variable_scope('deconv3') as scope:
            dekernel3 = self._create_weights([4, 7, 32, 64])
            output_shape3 = tf.stack([tf.shape(images)[0],50,81,32])
            deconv3 = self._create_deconv(deconv2, dekernel3, output_shape3)
            debias3 = self._create_bias([32])
            preactivation2 = tf.nn.bias_add(deconv3, debias3)#50*81*32
            deconv3 = tf.nn.elu(preactivation2, name=scope.name)
            #self._activation_summary(deconv3)
            
        with tf.variable_scope('deconv4') as scope:
            dekernel4 = self._create_weights([5, 7, 16, 32])
            output_shape4 = tf.stack([tf.shape(images)[0],103,167,16])
            deconv4 = self._create_deconv(deconv3, dekernel4, output_shape4)
            debias4 = self._create_bias([16])
            preactivation2 = tf.nn.bias_add(deconv4, debias4)#103*167*16
            deconv4 = tf.nn.elu(preactivation2, name=scope.name)
            #self._activation_summary(deconv4)
        
        with tf.variable_scope('deconv5') as scope:
            dekernel5 = self._create_weights([4, 8, 2, 16])
            output_shape5 = tf.stack([tf.shape(images)[0],208,340,2])
            deconv5 = self._create_deconv(deconv4, dekernel5, output_shape5)
            debias5 = self._create_bias([2])
            preactivation2 = tf.nn.bias_add(deconv5, debias5)#208*340*2
            deconv5 = tf.nn.elu(preactivation2, name=scope.name)
            # #self._activation_summary(deconv5)
            
        with tf.variable_scope('conv_1by1') as scope:
            kernel_f = self._create_weights([1, 1, 2, 2])
            conv_f = self._create_conv2d_V(deconv5, kernel_f)
            bias_f = self._create_bias([2])
            preactivation = tf.nn.bias_add(conv_f, bias_f)
            # conv_f = tf.nn.elu(preactivation, name=scope.name)#208*340*2
            #self._activation_summary(conv_1by1)
                       
        # with tf.variable_scope('fc1') as scope:
            
            reshape = tf.reshape(preactivation,shape=[-1,208,340,2])
            
            # fc = tf.layers.flatten(reshape)
            # fc = tf.layers.flatten(preactivation)
            # fc1 = tf.layers.dense(inputs=fc,units=self._num_labels,
            #                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),activation="leaky_relu")
            # fc2 = tf.layers.dense(inputs=fc1,units=3,activation=None)
            # fc2 = tf.reshape(fc2,shape=[-1,3,1])
            # self._activation_summary(fc1)
        return reshape

    def train(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, y_pre, ys,input_mask):#修修
        with tf.variable_scope('loss') as scope:
            
            y_pre1 = y_pre*input_mask
            ys1 = ys*input_mask
            cross_entropy = tf.cast(tf.subtract(y_pre1,ys1),tf.float32)
            cost = tf.reduce_mean(tf.square(cross_entropy), name=scope.name)
            tf.add_to_collection("losses",cost)
            loss = tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('cost', loss)

        return cost

    def accuracy(self, logits, ys,input_mask):
        with tf.variable_scope('accuracy') as scope:
            
            logits1 = logits*input_mask
            ys1 = ys*input_mask
          
            accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.subtract(logits1,ys1)),tf.float32),
                                      name=scope.name)
            #tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _create_conv2d_V2(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 2, 2, 1],
                            padding='VALID')
    
    def _create_conv2d_V(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='VALID')
    
#    def _create_conv2d_S(self, x, W):
#        return tf.nn.conv2d(input=x,
#                            filter=W,
#                            strides=[1, 1, 1, 1],
#                            padding='SAME')

#    def _create_max_pool_2x2(self, input):
#        return tf.nn.max_pool(value=input,
#                              ksize=[1, 2, 2, 1],
#                              strides=[1, 2, 2, 1],
#                              padding='VALID')
        
    def _create_deconv(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(value=x,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1,2,2,1],
                                      padding='VALID')
        
    def _create_deconv1(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(value=x,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1,1,1,1],
                                      padding='VALID')

    def _create_weights(self, shape):
        Var = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32))
        tf.add_to_collection("losses",tf.keras.regularizers.l2(l=0.0001)(Var))
        return Var

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))

    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    def get_data(num):
        prep = np.load('testdata/SDF.npy')[num]
        label = np.load('testdata/label.npy')[num]
        mask = np.load('testdata/mask.npy')[num]
        return prep, label, mask
    
    def get_US(num):
        Mean = np.load('testdata/Mean_label.npy')[num]
        S = np.load('testdata/S_label.npy')[num]
        return Mean, S
    
    def get_xy(num):
        d = np.load('testdata/airfoil.npy')[num]
        x1 = d[0:31,0]; y1 = d[0:31,1]
        x2 = d[30:61,0]; y2 = d[30:61,1]
        x3 = x2[::-1]; y3 = y2[::-1]
        return x1,y1,x2,y2,x3,y3