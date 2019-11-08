import os
import matplotlib.pyplot as plt
import itertools
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dataset path is here,include train and test.
dst_path = 'Y:\\catdog'

# your model file and file path
# model_file = "Y:\\catdog\\model\\keras_iv3_ft.h5"
model_file = "Y:\\catdog\\2019-10-31_inceptionResNet\\model\\keras_iv3_ft.h5"

# Dataset\
test_dir = os.path.join(dst_path, 'test_small')

# batch size
batch_size = 64

# load model
model = load_model(model_file)

# generator image
test_datagen = ImageDataGenerator(
	rescale=1. / 255
	)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle = False
    # class_mode='binary'
    )

# Part1.eval your model
def data_eval():
	test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples / batch_size,verbose=1)
	# test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size,verbose=1)
	print("this is the h5 file name: " + model_file)
	print("the test_loss is : " + str(test_loss))
	print("the test_acc is : " + str(test_acc))
	print('test acc: %.4f%%' % (test_acc * 100))


# Part2.build sonfusion_matrix and draw it

# def plot_sonfusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # 归一化
	plt.imshow(cm, interpolation='nearest', cmap=cmap) # 特定的窗口显示图像
	plt.title(title) # 图像标题
	plt.colorbar()
	# tick_marks = np.arange(len(classes))
	# plt.xticks(tick_marks, classes, rotation=45) # 标签印在X轴坐标上
	# plt.yticks(tick_marks, classes) # 标签印在Y轴坐标上
	num_local = np.array(range(len(classes)))
	plt.xticks(num_local, classes, rotation=90)    # 将标签印在x轴坐标上
	plt.yticks(num_local, classes)    # 将标签印在y轴坐标上


	thresh = cm.max() / 2.0
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, 
		# cm[i, j], 
		'{:.2f}'.format(cm[i, j]),
		horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predict label')
	plt.show()

# def plot_confuse(model, x_val, y_val):
def plot_confuse(model_file):
    # predictions = model.predict_classes(x_val) # model.predict_classes只能用于序列模型，不能用于函数式模型
    # predictions = model.predict(x_val)
    predict = model.predict_generator(test_generator,verbose=1)
    predict_y = np.argmax(predict,axis=1)
    # truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    # truelabel = predict_y   # 将one-hot转化为label
    true_y = test_generator.classes
    conf_mat = confusion_matrix(y_true=true_y, y_pred=predict_y)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(true_y)+1))
    # print("----------------conf_mat and cmap")
    # print(conf_mat)
    # print(range(np.max(true_y)+1))



if __name__=="__main__":
	# 将predict中得到的数据转化为混淆矩阵需要的参数
	# predict = model.predict_generator(test_generator,verbose=1)
	# print(predict)
	# predict_y = np.argmax(predict,axis=1) # 将predict后的数组转化为实际预测的类（即预测结果）
	# print(predict_y)
	# true_y = test_generator.classes #验证集中就有true labe的标签了
	# print(true_y)
	print("here is data eval result: ")
	data_eval()


	print("here is the matrix")
	# plot_confuse(model_file, predict_y, true_y)
	plot_confuse(model_file)