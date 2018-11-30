# CNN practice code with Dog and Cat image data from kaggle
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

train_dir = 'all/train'
test_dir = 'all/test'

# define image size and learning rate

img_size = 50
lr = 1e-3

# read trianing data
train_data = []
for pic in tqdm(os.listdir(train_dir)) :
    if not pic.endswith('.jpg') :
        continue

    img_class = pic.split('.')[0]
    if img_class == 'cat':
        lable = [1, 0]
    if img_class == 'dog':
        lable = [0, 1]
        
    path = os.path.join(train_dir, pic)
    pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pic = cv2.resize(pic, (img_size, img_size))
        
    train_data.append([np.array(pic), np.array(label)])

# process test data
test_data = []
for pic in tqdm(os.listdir(test_dir)) :
    if not pic.endswith('.jpg') :
        continue

    img_num = pic.split('.')[0]

    path = os.path.join(test_dir, pic)
    pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pic = cv2.resize(pic, (img_size, img_size))
        
    test_data.append([np.array(pic), img_num])

# shuffle train and test dataset
shuffle(train_data)
shuffle(test_data)


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.reset_default_graph()

# input layers
net = input_data(shape = [None, img_size, img_size, 1], name = 'input')

# will use 5 CNN and 3 maxpooling according to Alexnet
net = conv_2d(net, 96, 11, strides = 4 activation = 'relu')
net = max_pool_2d(net, 3, strides = 2)

net = conv_2d(net, 256, 5, activation = 'relu')
net = max_pool_2d(net, 3, strides = 2)

net = conv_2d(net, 384, 3, activation = 'relu')

net = conv_2d(net, 384, 3, activation = 'relu')

net = conv_2d(net, 256, 3, activation = 'relu')
net = max_pool_2d(net, 3, strides = 2)

# 3 fully connected layers
net = fully_connected(net, 4096, activation = 'tanh')
net = dropout(net, 0.5)

net = fully_connected(net, 4096, activation = 'tanh')
net = dropout(net, 0.5)

net = fully_connected(net, 17, activation = 'softmax')
net = regression(net, optimizer = 'adam', learning_rate = lr, loss = 'categorical_crossentropy', name = 'targets')

model = tflearn.DNN(net, tensorboard_dir = 'log')
train = train_data[:-500]
vali = train_data[-500:]

X = np.array([i[0] for i in train], dtype = np.float64).reshape(-1, img_size, img_size, 1)
Y = np.array([i[1] for i in train], dtype = np.float64)

X_vali = np.array([i[0] for i in vali], dtype = np.float64).reshape(-1, img_size, img_size, 1)
Y_vali = np.array([i[1] for i in vali], dtype = np.float64)

model.fit({'input': X}, {'targets': Y}, n_epoch = 1000, validation_set = ({'input': X_vali}, {'targets': Y_vali}), snapshot_step = 200, show_metric = True, batch_size = 64, snapshot_step = 200, snapshot_epoch = False, run_id = 'model')


# visualize the prediction
for i, data in enumerate(test_data[:9]):
    img_data = data[0]
    
    orig = img_data
    test = img_data.reshape(img_size, img_size, 1)
    
    model_out = model.predict([test])[0]

    if np.argmax(model_out) == 1:
        label = 'Dog'
    else:
        label = 'Cat'

    y = plt.figure().add_subplot(3, 3, i + 1)
    y.imshow(orig)
    plt.title(label)

plt.tight_layout()
plt.show()
