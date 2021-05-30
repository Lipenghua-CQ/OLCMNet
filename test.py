import tensorflow as tf
from olcmNet import OLCMNet
import cv2
import numpy as np
from tqdm import tqdm
import glob

one_hot = tf.keras.utils.to_categorical
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


def load_raw_image(data_base_path, target_shape):
    print('---Start loading image data---')
    h, w = target_shape[:-1]
    images = []
    labels = []
    all_path = []
    file_names = [data_base_path + 'c' + str(i) + '/*.jpg' for i in range(10)]
    print(file_names)
    for file in file_names:
        if 'c0' == file.split('/')[1]:
            label = 0
        elif 'c1' == file.split('/')[1]:
            label = 1
        elif 'c2' == file.split('/')[1]:
            label = 2
        elif 'c3' == file.split('/')[1]:
            label = 3
        elif 'c4' == file.split('/')[1]:
            label = 4
        elif 'c5' == file.split('/')[1]:
            label = 5
        elif 'c6' == file.split('/')[1]:
            label = 6
        elif 'c7' == file.split('/')[1]:
            label = 7
        elif 'c8' == file.split('/')[1]:
            label = 8
        elif 'c9' == file.split('/')[1]:
            label = 9
        paths = glob.glob(file)
        all_path.extend(paths)
        for image in tqdm(paths):
            img = cv2.imread(image)
            img = cv2.resize(img, (w, h))
            images.append(img / 255)
            labels.append(label)

    images = np.asarray(images, dtype=np.float32)
    labels = one_hot(labels, num_classes=10)
    # shuffle = np.arange(images.shape[0])
    # np.random.shuffle(shuffle)
    print('---Image data loaded---')
    return images, labels, all_path


def evaluate(data_base_path, target_shape, model_dir):
    lr = 0.1
    SGD = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=0.0001)

    images, labels, path = load_raw_image(data_base_path, target_shape)

    model = OLCMNet(input_shape=target_shape, num_classes=10, width_multiplier=1.0, )
    model.load_weights(model_dir + 'OLCMNet.h5')
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])
    model.evaluate(x=images, y=labels)



if __name__ == '__main__':
    evaluate('label_test/', (128, 256, 3), 'models/')
