import matplotlib.pyplot as plt

import os
import itertools
import cv2

import numpy as np
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# dst_path = 'Y:\\endonew_9_2'
# model_file = "Y:\\endonew_9_2\\ckpt\\tl_weights.0072-0.6124.h5"
# model_file = "Y:\\endonew_9_2\\2019-09-30_cleandata_1000epochs\\ckpt\\ft_weights.0533-0.7538.h5"
# model_file = "D:\\work\\git\\myself\\DeepLearningSandbox\\transfer_learning\\ft_0925_weights.0291-0.7957_4class.h5"
dst_path = 'Y:\\gucyTest'
model_file = "Y:\\gucyTest\\ckpt\\ft_weights.0549-0.7239.h5"
# model_file = "Y:\\endonew_9_2\\model\\keras_iv3_ft.h5"
test_dir = os.path.join(dst_path, 'test')

batch_size = 128

model = load_model(model_file)

test_datagen = ImageDataGenerator(
	rescale=1. / 255
	)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    # class_mode='binary'
    )

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples / batch_size)
print(test_loss)
print(test_acc)
print("this is the h5 file name: " + model_file)
print('test acc: %.4f%%' % (test_acc * 100))


print("--------------------------------------cut line--------------------------------")
test_loss, test_acc = model.predict_generator(test_generator, steps=test_generator.samples / batch_size)
print('test acc: %.4f%%' % (test_acc * 100))