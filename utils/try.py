#划分训练集和测试集
import os
import random
import shutil
import numpy as np
import keras
import math
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from torch import batch_norm

def Split():
    path = './Dataset_2/Train/'
    dirs = []
    split_percentage = 0.2

    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            fullpath = os.path.join(dirpath, dirname)
            fileCount = len([name for name in os.listdir(fullpath) if os.path.isfile(os.path.join(fullpath, name))])
            files = os.listdir(fullpath)
            for index in range((int)(split_percentage * fileCount)):
                newIndex = random.randint(0, fileCount - 1)
                fullFilePath = os.path.join(fullpath, files[newIndex])
                newFullFilePath = fullFilePath.replace('Train', 'Final_Validation')
                base_new_path = os.path.dirname(newFullFilePath)
                if not os.path.exists(base_new_path):
                    os.makedirs(base_new_path)
                # move the file
                try:
                    shutil.move(fullFilePath, newFullFilePath)
                except IOError as error:
                    print('skip moving from %s => %s' % (fullFilePath, newFullFilePath))

def train ():

    train_set_base_dir = './Dataset_2/Train'
    validation_set_base_dir = './Dataset_2/Final_Validation'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_data_generator = train_datagen.flow_from_directory(
        directory=train_set_base_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical')


    validation_datagen = ImageDataGenerator(
        rescale=1. /255
    )#图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；值将在执行其他处理前乘到整个图像上，
    #我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。

    validation_data_generator = validation_datagden.flow_from_directory(
        directory=validation_set_base_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical'
    )

    # define a simple CNN network
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

    model = Sequential()

    # add Con2D layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='elu', input_shape=(48, 48, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    # flatten
    model.add(Flatten())#压缩成一维

    # dropOut layer
    model.add(Dropout(0.2))#丢弃率0.2，防止过拟合

    # add one simple layer for classification
    model.add(Dense(units=512, activation='elu'))#全连接层

    # add output layer
    model.add(Dense(units=43, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    #model.compile(loss = 损失函数，optimizer = 优化器，metrics = ["准确率”])
    # print model info
    model.summary()
    json_str = model.to_json()#保存模型
    print(json_str)
    # fit_generator to fill in the dataset
    history = model.fit_generator(
        generator=train_data_generator,
        steps_per_epoch=100,#图片总张数除以batchsize
        epochs=60,
        validation_data=validation_data_generator,
        validation_steps=50)

    # train done, save the models
    model.save('C:/test/WorkingLogs/20181214/traffic_signs.h5')

    # plot the roc curve
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def test():

    test_image=[]
    test_lable=[]
    x=''
    base_path = './Dataset_2/'

    with open('./Dataset_2/Test.csv','r',newline='') as file:
        header = file.readline()
        header = header.strip()
        header_list = header.split(',')
        # print(header_list)

        for row in file.readlines():
            row_data = row.split(',')
            x=row_data[7]
            x=base_path+x
            x=x.strip('\n')
            test_lable.append(int(row_data[6]))
            test = Image.open(x)
            test = test.resize((48,48),Image.ANTIALIAS)
            test = np.array(test)
            test_image.append(test)

    test_data = np.array(test_image)
    test_lable = np.array(test_lable)
    # print(test_data)
    # print(test_lable)

    #标签进行one-hot编码
    labels = test_lable
    one_hot_labels = tf.one_hot(indices=labels,depth=43, on_value=1, off_value=0, axis=-1, dtype=tf.int32, name="one-hot")
    # print(one_hot_labels.shape)

    test_datagen = ImageDataGenerator(
        rescale=1. /255
    )

    test_data_generator = test_datagen.flow(
        x=test_data,
        y=one_hot_labels,
        #target_size=(48, 48),
        batch_size=32
        #class_mode='categorical'
    )

    #import keras
    # new_model = keras.models.load_model('C:/test/WorkingLogs/20181214/traffic_signs.h5')
    #test_loss, test_acc = new_model.evaluate(test_image, test_lable)
    #print('\nTest accuracy:', test_acc)
    #附加一个 softmax 层，将 logits 转换成更容易理解的概率
    # probability_model = tf.keras.Sequential([new_model, 
    #                                         tf.keras.layers.Softmax()])

    #test_loss, test_acc = new_model.evaluate_generator(validation_data_generator)
    #print('\nTest accuracy:', test_acc)
    # new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # new_model.summary()

    #test评估
    # scores2 = new_model.evaluate(test_data,one_hot_labels, verbose=2)
    # print(scores2)



if __name__ == "__main__":
    # split()
    train()
    # test()

