# keras 迁移训练 用inceptionV3
# 首先import keras inception 权重文件
from keras.applications.inception_v3 import InceptionV3,preprocess_input
# 这里同时放inception resnet v2
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

import os
# use gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# myself start，数据路径
train_dir = "./train"
test_dir = "./test"
validation_dir = "./validation"
nb_class = 2
Freeze_Layer = 1

# 在新的数据集上训练几代
# 也就是转移训练、迁移学习
initial_epoch   = 0
final_epoch     = 2

# 数据增强
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    # 应用 ZCA 白化
    zca_whitening=True,
    # ZCA 白化的 epsilon 值
    zca_epsilon=1e-06,
    preprocessing_function=preprocess_input,
    rotation_range = 90, #随机旋转图像15度
    width_shift_range = 0.1, #随机改变宽度，设置为6表示-3～+3，下一项相同
    height_shift_range = 0.1,
    # shear_range = 0.2, #剪切强度，逆时针方向剪切
    zoom_range = 0.2, #随机缩放范围，正负20%
    horizontal_flip = False, #随机垂直翻转
    vertical_flip = False #因内窥镜成像图始终保持在右侧，左侧为黑底文字，故此处不做水平翻转
    )
val_datagen = ImageDataGenerator(
	# rescale=1./255,
    # 应用 ZCA 白化
    zca_whitening=True,
    # ZCA 白化的 epsilon 值
    zca_epsilon=1e-06,
    preprocessing_function=preprocess_input,
    rotation_range = 90, #随机旋转图像15度
    width_shift_range = 0.1, #随机改变宽度，设置为6表示-3～+3，下一项相同
    height_shift_range = 0.1,
    # shear_range = 0.2, #剪切强度，逆时针方向剪切
    zoom_range = 0.2, #随机随机缩放范围，正负20%
    horizontal_flip = False, #随机垂直翻转
    vertical_flip = False #因内窥镜成像图始终保持在右侧，左侧为黑底文字，故此处不做水平翻转
    )

# 显示数据量
train_generator = train_datagen.flow_from_directory(
								directory = './train',
                                target_size = (299,299), #Inception V3规定大小是299
                                batch_size = 256 , # shuffle默认是true，所以此处只指定batch
                                shuffle = True, # 训练图片混洗
                                # save_to_dir = "aug/train", # 保存增强后的图片
                                # save_prefix = "augpic",  # 保存文件前缀
                                # save_format = "jpeg" # jpg格式 
                                )
val_generator = val_datagen.flow_from_directory(
								directory = './test',
                                target_size = (299,299),
                                batch_size = 256 , 
                                shuffle = False, # 验证图片不混洗
                                # save_to_dir = "aug/etst/", # 保存增强后的图片
                                # save_prefix = "augpic",  # 保存文件前缀
                                # save_format = "jpeg" # jpg格式 
                                )

# 训练的每epoch训练多少次
tl_steo_per_epoch = train_generator.n // train_generator.batch_size
ft_steo_per_epoch = train_generator.n // train_generator.batch_size

tl_val_steps = val_generator.n // val_generator.batch_size
ft_val_steps = val_generator.n // val_generator.batch_size
# myself end



# 构建不带分类器的预训练模型
# imagenet表示下载已经训练好的参数，include_top是true的话输出1000个节点的全连接层，false则去掉顶层，输出一个8 8 2048的tensor
base_model = InceptionV3(weights='imagenet', include_top=False)
# switch inception and inceptionResnetV2.choice a base model to training
# base_model = InceptionResNetV2(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x) # for InceptionV3
# x = Dense(768, activation='relu')(x) # for InceptionResNet

# 添加一个分类器，假设我们有200个类
# 此处改为13类
predictions = Dense(nb_class, activation='softmax')(x)

# 构建我们需要训练的完整模型
# 如需继续训练，请把第二行注释取消，并load正确的断点
# 单GPU版本
model = Model(inputs=base_model.input, outputs=predictions)
# model = load_model('./model/keras_iv3_tl.h5')

# 多GPU版本
# model = multi_gpu_model(model, gpus=2)



# 画模型结构图
# plot_model(model, 'tl.png')

# myself start，这里设置callbacks函数包含的内容，如tensorboard、csvlog等，用于观测训练过程中的数据变化
terminate_on_nan = TerminateOnNaN() # 当遇到NaN loss时会停止训练

model_checkpoint_tl = ModelCheckpoint(
	filepath = './ckpt/tl_weights.{epoch:02d}-{val_acc:.4f}.h5', # 保存路径和文件名中包含轮数及当时的val_acc值
	monitor = 'val_acc' , # 监视的数据
	verbose = 1, #详细模式
	save_best_only = True, #被监测数据的最佳模型不会被覆盖
	mode = 'auto', #自动判断
	save_weights_only = False, # 不止保存权重，而是保存完整模型
	period = 1# 每轮监控一次，不间断
	)
model_checkpoint_ft = ModelCheckpoint(
	filepath = './ckpt/ft_weights.{epoch:02d}-{val_acc:.4f}.h5', # 保存路径和文件名中包含轮数及当时的val_acc值
	monitor = 'val_acc' , # 监视的数据
	verbose = 1, #详细模式
	save_best_only = True, #被监测数据的最佳模型不会被覆盖
	mode = 'auto', #自动判断
	save_weights_only = False, # 不止保存权重，而是保存完整模型
	period = 1# 每轮监控一次，不间断
	)

csv_logger_tl = CSVLogger(
	filename = './csv/CSVLogger_tl.csv', # 训练轮结果数据流到 csv 文件
	separator=',',
	append = True
	)

csv_logger_ft = CSVLogger(
	filename = './csv/CSVLogger_ft.csv', # 训练轮结果数据流到 csv 文件
	separator=',',
	append = False
	)

# 学习速率变化
# def lr_schedule(epoch):
#     if epoch < 80:
#         return 0.001
#     elif epoch < 100:
#         return 0.0001
#     else:
#         return 0.00001

# learning_rate_scheduler = LearningRateScheduler(
# 	schedule=lr_schedule,
# 	verbose=1
# 	)

# 记录transfer learning的tensorboard
tensor_board_tl = TensorBoard(
	log_dir='./tb_tl', 
	histogram_freq=0
	)
# 记录fine tune的tensorboard
tensor_board_ft = TensorBoard(
	log_dir='./tb_ft', 
	histogram_freq=0
	)

callbacks_tl = [terminate_on_nan ,
model_checkpoint_tl ,
csv_logger_tl ,
tensor_board_tl
]

callbacks_ft = [terminate_on_nan ,
model_checkpoint_ft ,
csv_logger_ft ,
tensor_board_ft
]

# 以下为transfer learning部分
# 锁住所有卷积，其实等于只训练了最顶部的全连接层

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(
	optimizer=SGD(lr=0.0001, momentum=0.9), 
	# optimizer='RMSprop',
	loss='categorical_crossentropy',
	metrics=['accuracy']
	)

transfer_learning_train = model.fit_generator(
	generator = train_generator,
	steps_per_epoch = tl_steo_per_epoch ,
	initial_epoch = initial_epoch,
	epochs = final_epoch, # 训练轮数
	validation_data = val_generator,
	validation_steps = tl_val_steps,
	class_weight = 'auto',
	callbacks = callbacks_tl #回调函数
	)
# 训练后保存最后的模型
model.save('./model/keras_iv3_tl.h5')



# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。

# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
# 这步就停下来吧，只是为了看网络结构，确定底部卷积层在哪
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)


# 以下为fine tune部分
# 依旧锁住底部的卷积层，重新训练其他网络层

# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
for layer in model.layers[:Freeze_Layer]:
   layer.trainable = False
for layer in model.layers[Freeze_Layer:]:
   layer.trainable = True

# 我们需要重新编译模型，才能使上面的修改生效
# 让我们设置一个很低的学习率，使用 SGD 来微调

model.compile(
	optimizer=SGD(lr=0.00001, momentum=0.9), 
	loss='categorical_crossentropy',
	metrics=['accuracy']
	)

# 我们继续训练模型，这次我们训练最后两个 Inception block
# 和两个全连接层
finetune_train = model.fit_generator(
	generator = train_generator,
	steps_per_epoch = ft_steo_per_epoch ,
	epochs = final_epoch, # 训练轮数
	validation_data = val_generator,
	validation_steps = ft_val_steps,
	class_weight = 'auto',
	callbacks = callbacks_ft #回调函数
	)

# 训练结束后保存最终模型
model.save('./model/keras_iv3_ft.h5')

# model.summary()

# plt可视化训练过程
# def plot_training_tl(transfer_learning_train):
#   acc = transfer_learning_train.history['acc']
#   val_acc = transfer_learning_train.history['val_acc']
#   loss = transfer_learning_train.history['loss']
#   val_loss = transfer_learning_train.history['val_loss']
#   epochs = range(len(acc))

#   plt.plot(epochs, acc, 'r.')
#   plt.plot(epochs, val_acc, 'r')
#   plt.title('Training and validation accuracy')

#   plt.figure()
#   plt.plot(epochs, loss, 'r.')
#   plt.plot(epochs, val_loss, 'r-')
#   plt.title('Training and validation loss')

#   plt.show()

# def plot_training_ft(finetune_train):
#   acc = finetune_train.history['acc']
#   val_acc = finetune_train.history['val_acc']
#   loss = finetune_train.history['loss']
#   val_loss = finetune_train.history['val_loss']
#   epochs = range(len(acc))

#   plt.plot(epochs, acc, 'r.')
#   plt.plot(epochs, val_acc, 'r')
#   plt.title('Training and validation accuracy')

#   plt.figure()
#   plt.plot(epochs, loss, 'r.')
#   plt.plot(epochs, val_loss, 'r-')
#   plt.title('Training and validation loss')

#   plt.show()

# plot_training_tl(transfer_learning_train)
# plot_training_ft(finetune_train)
