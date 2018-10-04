import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

train_dir = 'train'
test_dir = 'test'
img_size = 50
lr = 1e-3 #learning rate

def label_img(img) :
    # cat.12.jpg
    word_label = img.split('.')[-3]
    if word_label == 'cat' :
        return [1, 0]
    elif word_label == 'dog' :
        return [0, 1]

# read trianing data
def create_train_data() :
    training_data = []
    for img in tqdm(os.listdir(train_dir)) :
        if not img.endswith('.jpg') :
            continue

        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    return training_data

train_data = create_train_data()

# process test data
def process_test_data() :
    testing_data = []
    for img in tqdm(os.listdir(test_dir)) :
        if not img.endswith('.jpg') :
            continue

        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    return testing_data

test_data = process_test_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.reset_default_graph()

# input layers
convnet = input_data(shape = [None, img_size, img_size, 1], name = 'input')

# will use 5 CNN and 5 maxpooling
convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

# 2 fully connected layers
convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation = 'softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate = lr, loss = 'categorical_crossentropy', name = 'targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')
train = train_data[:-500]
vali = train_data[-500:]

X = np.array([i[0] for i in train], dtype = np.float64).reshape(-1, img_size, img_size, 1)
Y = np.array([i[1] for i in train], dtype = np.float64)

X_vali = np.array([i[0] for i in vali], dtype = np.float64).reshape(-1, img_size, img_size, 1)
Y_vali = np.array([i[1] for i in vali], dtype = np.float64)

# training
model.fit({'input': X}, {'targets': Y}, n_epoch = 3, validation_set = ({'input': X_vali}, {'targets': Y_vali}), snapshot_step = 500, show_metric = True, run_id = 'model')

# reading test test_data
test_data = process_test_data()

# output some prediction on the test data
fig = plt.figure()
for num, data in enumerate(test_data[:16]) :
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(4, 4, num + 1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1 :
        label = 'Dog'

    else :
        label = 'Cat'

    y.imshow(orig, cmap = 'gray')
    plt.title(label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
