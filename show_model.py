# keras 迁移训练 用inceptionV3
# 首先import keras inception 权重文件
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import multi_gpu_model
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TerminateOnNaN ,TensorBoard
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# model = load_model('./ft_0925_weights.0291-0.7957.h5')
# model = InceptionResNetV2(weights='imagenet', include_top=True)
model = InceptionV3(weights='imagenet', include_top=True)

# 添加全局平均池化层
# x = model.output
# x = GlobalAveragePooling2D()(x)

# # # 添加一个全连接层
# x = Dense(768, activation='relu')(x)

# # # 添加一个分类器，假设我们有200个类
# # # 此处改为13类
# predictions = Dense(3, activation='softmax', name="final")(x)

# # # 构建我们需要训练的完整模型
# # # 如需继续训练，请把第二行注释取消，并load正确的断点
# # # 单GPU版本
# model = Model(inputs=model.input, outputs=predictions)

# print(model)


# for layer in model.layers[:4]:
#    layer.trainable = False
# for layer in model.layers[4:]:
#    layer.trainable = True

plot_model(model, 'InceptionV3.png')
for i, layer in enumerate(model.layers):
    print(i, layer.name)
    print(i, layer.trainable)

print("==============================================================")
model.summary()