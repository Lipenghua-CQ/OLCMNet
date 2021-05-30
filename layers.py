import tensorflow as tf

class ReLU6(tf.keras.layers.Layer):
    def __init__(self, name="ReLU6"):
        super().__init__(name=name)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6")

    def call(self, input):
        return self.relu6(input)


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, name="HardSigmoid"):
        super().__init__(name=name)
        self.relu6 = ReLU6()

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0


class HardSwish(tf.keras.layers.Layer):
    def __init__(self, name="HardSwish"):
        super().__init__(name=name)
        self.hard_sigmoid = HardSigmoid()

    def call(self, input):
        return input * self.hard_sigmoid(input)


class Squeeze(tf.keras.layers.Layer):
    """Squeeze the second and third dimensions of given tensor.
    (batch, 1, 1, channels) -> (batch, channels)
    """
    def __init__(self):
        super().__init__(name="Squeeze")

    def call(self, input):
        x = tf.keras.backend.squeeze(input, 1)
        x = tf.keras.backend.squeeze(x, 1)
        return x



def get_shape(x):
    return tf.keras.backend.int_shape(x)


def globalAvergePooling2D(x):
    pool_size = get_shape(x)[1:3]
    return tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)


def batchNormalization(x):
    return tf.keras.layers.BatchNormalization(momentum=0.99)(x)


def channel_split(x, name=''):
    # equipartition
    in_channles = get_shape(x)[-1]
    ip = in_channles // 2
    c_hat = tf.keras.layers.Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = tf.keras.layers.Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def convNormAct(x, filters, name:str,k_sizes=3, stride=1, padding='VALID',
                norm_layer=None, act_layer="relu", use_bias='True', l2_reg=1e-5):

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=k_sizes, strides=stride,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                               use_bias=use_bias, padding=padding, name=name)(x)
    if norm_layer:
        x = batchNormalization(x)

    _available_activation = {
        "relu": tf.keras.layers.ReLU(name=name+'/relu'),
        "relu6": ReLU6(name=name+'/relu6'),
        "hswish": HardSwish(name=name+'hswish'),
        "hsigmoid": HardSigmoid(name=name+'hsigmoid'),
        "softmax": tf.keras.layers.Softmax(name='softmax'),
    }
    if act_layer:
        x = _available_activation[act_layer](x)

    return x

def seBottleneck(x, name, reduction=4, l2_reg=0.01):
    input_channel = get_shape(x)[3]
    gap = globalAvergePooling2D(x)
    conv1 = convNormAct(gap, input_channel // reduction, k_sizes=1, norm_layer=None, act_layer='relu', use_bias=False,
                        l2_reg=l2_reg, name=name + '/conv1')
    conv2 = convNormAct(conv1, input_channel, k_sizes=1, norm_layer=None, act_layer='hsigmoid', use_bias=False,
                        l2_reg=l2_reg, name=name + '/conv2')

    return tf.keras.layers.Multiply()([x, conv2])

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v
