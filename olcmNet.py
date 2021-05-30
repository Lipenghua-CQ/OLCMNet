from layers import *
from keras.utils.vis_utils import plot_model

# OLCM block
def olcm_block(x, out_channels, exp_channels, k_size, stride, use_se, act_layer, name,l2_reg=1e-5):
    input_channel = get_shape(x)[3]

    # path 1 : low 1
    expand1 = convNormAct(x, filters=exp_channels // 2, k_sizes=1, norm_layer='bn', act_layer=act_layer, use_bias=False,
                          l2_reg=l2_reg, name=name + '/expand1')
    expand1 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), name=name+'avg')(expand1)

    depthwise1 = tf.keras.layers.DepthwiseConv2D(kernel_size=k_size, strides=stride, padding="SAME",
                                                 depthwise_regularizer=tf.keras.regularizers.l2(l2_reg), use_bias=False,
                                                 name=name + '/depthwise1')(expand1)
    bn1 = batchNormalization(depthwise1)

    _available_activation = {
        "relu": tf.keras.layers.ReLU(name=name + "/Depthwise1/ReLU"),
        "hswish": HardSwish(name=name + "/Depthwise1/HardSwish"),
    }
    act1 = _available_activation[act_layer](bn1)

    act1 = tf.keras.layers.UpSampling2D((2, 2))(act1)

    # path2 : high
    expand2 = convNormAct(x, filters=exp_channels // 2, k_sizes=1, norm_layer='bn', act_layer=act_layer, use_bias=False,
                          l2_reg=l2_reg, name=name + '/expand2')
    depthwise2 = tf.keras.layers.DepthwiseConv2D(kernel_size=k_size, strides=stride, padding="SAME",
                                                 depthwise_regularizer=tf.keras.regularizers.l2(l2_reg), use_bias=False,
                                                 name=name + '/depthwise2')(expand2)
    bn2 = batchNormalization(depthwise2)

    _available_activation = {
        "relu": tf.keras.layers.ReLU(name=name + "/Depthwise2/ReLU"),
        "hswish": HardSwish(name=name + "/Depthwise2/HardSwish"),
    }
    act2 = _available_activation[act_layer](bn2)

    # path 3 : low2
    depthwise3 = tf.keras.layers.DepthwiseConv2D(kernel_size=5, strides=stride, padding="SAME",
                                                 depthwise_regularizer=tf.keras.regularizers.l2(l2_reg), use_bias=False,
                                                 name=name + '/depthwise3')(expand1)
    bn3 = batchNormalization(depthwise3)

    _available_activation = {
        "relu": tf.keras.layers.ReLU(name=name + "/Depthwise3/ReLU"),
        "hswish": HardSwish(name=name + "/Depthwise3/HardSwish"),
    }
    act3 = _available_activation[act_layer](bn3)

    act3 = tf.keras.layers.UpSampling2D((2, 2))(act3)

    # concate
    act = tf.keras.layers.Concatenate(axis=-1)([act1, act2, act3])
    if use_se:
        act = seBottleneck(act, name=name + '/se')

    project = convNormAct(act, filters=out_channels, k_sizes=1, norm_layer='bn', act_layer=None, use_bias=False,
                          l2_reg=l2_reg, name=name + '/project')

    if stride == 1 and input_channel == out_channels:
        return tf.keras.layers.Add()([x, project])
    # use residual path
    # else:
    #     res = convNormAct(x, filters=out_channels, k_sizes=1, stride=stride, use_bias=False, name=name + '/res',
    #                       norm_layer='bn')
    #     return tf.keras.layers.Add()([res, project])
    return project

def lastStage(x, penultimate_channels, last_channels, num_class, l2_reg, name='last_stage'):
    conv1 = convNormAct(x, filters=penultimate_channels, k_sizes=1, stride=1, norm_layer='bn', act_layer='hswish',
                        use_bias=False, l2_reg=l2_reg, name=name+'/conv1')
    conv1 = seBottleneck(conv1, name=name+'conv1')
    gap = globalAvergePooling2D(conv1)
    conv2 = convNormAct(gap, filters=last_channels, k_sizes=1, norm_layer=None, act_layer='hswish', l2_reg=l2_reg, name=name+'/conv2')
    dropout = tf.keras.layers.Dropout(rate=0.2)(conv2)
    conv3 = convNormAct(dropout, filters=num_class, k_sizes=1, norm_layer=None, act_layer='softmax', l2_reg=l2_reg, name=name+'/conv3')
    pre = Squeeze()(conv3)

    return pre


def OLCMNet(input_shape, num_classes, width_multiplier=1, divisible_by=8, l2_reg=1e-5, batch_size=None):
    # First layer
    inputs = tf.keras.Input(shape=input_shape, name='inputs', batch_size=batch_size)
    x = convNormAct(inputs, filters=16, k_sizes=3, stride=2, padding='SAME', norm_layer='bn', act_layer='hswish', use_bias=False, l2_reg=l2_reg, name='first_layer')

    # bottleneck layers
    bneck_settings = [
                # k exp out  SE     NL       s
                [3, 72, 24, True, "hswish", 1],
                [3, 96, 40, True, "hswish", 2],
                [3, 240, 40, True, "hswish", 1],
                [3, 280, 80, True, "hswish", 2],
                [3, 672, 112, True, "hswish", 1],
        ]

    for i, (k, exp, out, se, nl, s) in enumerate(bneck_settings):
        # print(bneckes[i])
        out_channels = make_divisible(out*width_multiplier, divisible_by)
        exp_channels = make_divisible(exp*width_multiplier, divisible_by)
        # print(out_channels, exp_channels)
        x = olcm_block(x, out_channels=out_channels, exp_channels=exp_channels, k_size=k, stride=s, use_se=se, act_layer=nl, name='bneck'+str(i))

    # last stage
    penultimate_channels = make_divisible(960 * width_multiplier, divisible_by)
    last_channels = make_divisible(1_280 * width_multiplier, divisible_by)

    predict = lastStage(x, penultimate_channels=penultimate_channels, last_channels=last_channels, num_class=num_classes, l2_reg=l2_reg)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[predict])

    return model

if __name__ == '__main__':

    model = OLCMNet((128, 256, 3), 10, width_multiplier=1)
    model.load_weights('models\\OLCMNet.h5')
    # plot_model(model, show_shapes=True, to_file='olcmnet.png')
    model.summary()