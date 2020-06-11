# keras 迁移训练 用inceptionV3
# import tensorflow for multi_gpu
import tensorflow as tf
from keras.utils import multi_gpu_model
# 首先import keras inception 权重文件
from keras.applications.inception_v3 import InceptionV3,preprocess_input
# 这里同时放inception resnet v2
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
# VGG16
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TerminateOnNaN ,TensorBoard
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# use gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("=====================================this is preprocess_input")
print(preprocess_input)
print("=====================================this end preprocess_input")

# base trainning parameter start
train_dir = "./train"
test_dir = "./test"
validation_dir = "./eval"
nb_class = 20
Freeze_Layer = 1
GPU_numbers = 2
# if you only have a GPU，plz use single
# if you don't have a GPU,plz forget deep learning:D
train_mode = "single" 
# train_mode = "multi" 

# switch the keras official Applications
network_name = "InceptionResNetV2"
# network_name = "InceptionV3"
# network_name = "VGG16"
target_size = (299,299)
VGG_target_size = (224,224)
batch_size = 32
# trainning start and end epoch
initial_epoch   = 0
final_epoch     = 20
# fit_x = (-1,299,299,3)

# base trainning parameter end

# data augment
train_datagen = ImageDataGenerator(
    # ZCA whitening
    zca_whitening=True,
    # ZCA epsilon 
    zca_epsilon=1e-05,
    rescale=1./255,   # I dont know why,if use rescale,the tranning result is bad
    preprocessing_function=preprocess_input,
    # rotation_range = 180, # random rotation pic
    width_shift_range = 0.1, # random change pic width and hight
    height_shift_range = 0.1,
    # shear_range = 0.2, # random shear pic
    zoom_range = 0.2, # random zoom pic 
    horizontal_flip = False, # horizon flip pic 
    vertical_flip = False # vertical flip pic
    )
val_datagen = ImageDataGenerator(
    rescale=1./255, 
    # ZCA whitening
    # zca_whitening=True,
    # ZCA 白化的 epsilon 值
    # zca_epsilon=1e-06,
    preprocessing_function=preprocess_input,
    # rotation_range = 90, 
    # width_shift_range = 0.1, 
    # height_shift_range = 0.1,
    # shear_range = 0.2,
    # zoom_range = 0.2,
    horizontal_flip = False,
    vertical_flip = False
    )
# train_datagen.fit(fit_x, augment=True, rounds=1)
# val_datagen.fit(fit_x, augment=True, rounds=1)
# 调用训练及验证数据
train_generator = train_datagen.flow_from_directory(
								directory = train_dir,
                                target_size = target_size, #Inception V3 is 299
                                # target_size = VGG_target_size, # VGG is 244
                                # batch_size = batch_size , # 
                                shuffle = True, # train need shuffle
                                # save_to_dir = "aug/train", # save train pic after augmention，do NOT open this
                                # save_prefix = "augpic",  # file main name
                                # save_format = "jpeg" # file last name 
                                )
val_generator = val_datagen.flow_from_directory(
								directory = test_dir,
                                target_size = target_size, #Inception V3 is 299
                                # target_size = VGG_target_size, # VGG is 244
                                batch_size = batch_size , 
                                shuffle = False, # val pic need not shuffle
                                # save_to_dir = "aug/etst/", # 保存增强后的图片
                                # save_prefix = "augpic",  # 保存文件前缀
                                # save_format = "jpeg" # jpg格式 
                                )

# train and val per epoch
tl_steo_per_epoch = train_generator.n // train_generator.batch_size
ft_steo_per_epoch = train_generator.n // train_generator.batch_size

tl_val_steps = val_generator.n // val_generator.batch_size
ft_val_steps = val_generator.n // val_generator.batch_size

# model map
# plot_model(model, 'tl.png')

# callbacks,include tensorboard and csvlog.....
terminate_on_nan = TerminateOnNaN() # if NaN loss,stop train

model_checkpoint_tl = ModelCheckpoint(
	# filepath = './ckpt/tl_weights.{epoch:02d}-{val_acc:.4f}.h5', # save snapshot path and filename
	# filepath = './ckpt/tl_weights.{epoch:02d}-{val_accuracy:.4f}.h5', # keras 2.3.x, val_acc change to val_accuracy
	# filepath = './ckpt/tl_weights.{epoch:03d}-{val_acc:.4f}.h5', # keras 2.3.x, val_acc change to val_accuracy
	filepath = './ckpt/tl_weights.{epoch:03d}-{val_accuracy:.4f}.h5', # keras 2.3.x, val_acc change to val_accuracy
	monitor = 'val_accuracy' , # monitor val_accuracy, if you use keras 2.2.x ,please use "val_acc"
	# monitor = 'val_acc' , # monitor val_accuracy, if you use keras 2.2.x ,please use "val_acc"
	verbose = 1, # details mode
	save_best_only = True, # don't overwrite the best model
	mode = 'auto', # auto mode
	save_weights_only = False, # save full model
	period = 1 # monitior per 1 epoch 
	)
model_checkpoint_ft = ModelCheckpoint(
	# filepath = './ckpt/tl_weights.{epoch:02d}-{val_acc:.4f}.h5', # save snapshot path and filename
	# filepath = './ckpt/ft_weights.{epoch:02d}-{val_acc:.4f}.h5', # keras 2.3.x, val_acc change to val_accuracy
	filepath = './ckpt/ft_weights.{epoch:03d}-{val_accuracy:.4f}.h5', # keras 2.3.x, val_acc change to val_accuracy
	monitor = 'val_accuracy' , # monitor val_accuracy, if you use keras 2.2.x ,please use "val_acc"
	# monitor = 'val_acc' , # monitor val_accuracy, if you use keras 2.2.x ,please use "val_acc"
	verbose = 1, # details mode
	save_best_only = True, # don't overwrite the best model
	mode = 'auto', # auto mode
	save_weights_only = False, # save full model
	period = 1 # monitior per 1 epoch 
	)

csv_logger_tl = CSVLogger(
	filename = './csv/CSVLogger_tl.csv', # record a csv file
	separator=',',
	append = True
	)

csv_logger_ft = CSVLogger(
	filename = './csv/CSVLogger_ft.csv', # record a csv file
	separator=',',
	append = False
	)

# Dynamic lr
def scheduler(epoch):
    # 每隔X个epoch，学习率减小为原来的1/Y
    # X=9,Y=0.5
    if epoch % 7 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)

# under this commit is useless
# dynamic lr 
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

# record transfer learning to tensorboard
tensor_board_tl = TensorBoard(
	log_dir='./tb_tl', 
	histogram_freq=0
	)
# record fine tune to tensorboard
tensor_board_ft = TensorBoard(
	log_dir='./tb_ft', 
	histogram_freq=0
	)

callbacks_tl = [terminate_on_nan ,
model_checkpoint_tl ,
csv_logger_tl ,
tensor_board_tl,
reduce_lr
]

callbacks_ft = [terminate_on_nan ,
model_checkpoint_ft ,
csv_logger_ft ,
tensor_board_ft,
reduce_lr
]


# 
def network_mode(mode):
	print(mode)
	print("====================================================")
	global x
	global base_model
	global model
	if mode == "InceptionV3":
		# load Keras offical model
		base_model = InceptionV3(weights='imagenet', include_top=False)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(2048, activation='relu')(x) # InceptionV3 use 1024
		# predictions = Dense(nb_class, activation='softmax')(x)
		# model = Model(inputs=base_model.input, outputs=predictions)

	if mode == "InceptionResNetV2":
		base_model = InceptionResNetV2(weights='imagenet', include_top=False)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.5)(x)
		x = Dense(1536, activation='relu')(x) # inceptionResNetV2 use 768

	if mode == "VGG16":
		base_model = VGG16(weights='imagenet', include_top=False)
		x = base_model.output
		x = Flatten()(x)
		x = Dense(4096, activation='relu')(x)
		# x = Dropout(0.5)(x)
		x = Dense(4096, activation='relu')(x)

	
	predictions = Dense(nb_class, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)
	# if you need conutinue your train，please load your model snapshot here to overwrite the model
	# model = load_model('your h5 file path and filename')

	# forze all layer in base_model
	for layer in base_model.layers:
		layer.trainable = False

	model.compile(
		optimizer=SGD(lr=1e-03, momentum=0.9), 
		# optimizer='RMSprop',
		# optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False), 
		loss='categorical_crossentropy',
		metrics=['accuracy']
		)

	# for i, layer in enumerate(model.layers):
	# 	print(i, layer.name)
	# 	print(i, layer.trainable)
	return model

def train(train_mode):
	global parallel_model
	if train_mode == "multi":
		with tf.device('/cpu:0'):
			# input "InceptionResNetV2" or "InceptionV3" to choice an network
			model = network_mode(network_name)
		parallel_model = multi_gpu_model(model, gpus=GPU_numbers)
		work = parallel_model

	if train_mode == "single":
		model = network_mode(network_name)
		work = model

	# print(id(base_model))
	# print(id(model))
	# print(id(parallel_model))
	# print(id(work))
	# print("================================this is base_model/model/parallel_model/work in address of memory========================")

	# let work = model or parallel_model
	# so compile/fit work , in fact is compile/fit model
	# compile model
	work.compile(
		optimizer=SGD(lr=1e-03, momentum=0.9), 
		# optimizer='RMSprop',
		# optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False), 
		loss='categorical_crossentropy',
		metrics=['accuracy']
		)
	transfer_learning_train = work.fit_generator(
		generator = train_generator,
		steps_per_epoch = tl_steo_per_epoch ,
		initial_epoch = initial_epoch, # start epoch
		epochs = final_epoch, # end epoch
		validation_data = val_generator,
		validation_steps = tl_val_steps,
		class_weight = 'auto',
		callbacks = callbacks_tl # callbacks
		)
	# 训练后保存最后的模型
	if work == model:
		work.save('./model/' + network_name + '_tl.h5')
	else:
		model.save('./model/' + network_name + '_tl.h5') # for save multi gpu model,can not use parallel_model.save()
	

	# fine tune
	# freeze from layer 1 to Freeze_Layer , and train others
	for layer in model.layers[:Freeze_Layer]:
		layer.trainable = False
	for layer in model.layers[Freeze_Layer:]:
		layer.trainable = True

	work.compile(
		optimizer=SGD(lr=1e-3, momentum=0.9), 
		# optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False), 
		loss='categorical_crossentropy',
		metrics=['accuracy']
		)

	finetune_train = work.fit_generator(
		generator = train_generator,
		steps_per_epoch = ft_steo_per_epoch , # start epoch
		epochs = final_epoch, # end epoch
		validation_data = val_generator,
		validation_steps = ft_val_steps,
		class_weight = 'auto',
		callbacks = callbacks_ft #回调函数
		)
	if work == model:
		work.save('./model/' + network_name + '_ft.h5')
	else:
		model.save('./model/' + network_name + '_ft.h5') # for save multi gpu model,can not use 

# model.summary()

# plt train train_loss acc and val_loss acc
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

if __name__=="__main__":
	train(train_mode) 
