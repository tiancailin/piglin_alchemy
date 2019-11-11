import os
import time
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input


target_size = (299, 299) #fixed size for InceptionV3 architecture
batch_size = 64



def data_predict():
	# generator image
	test_datagen = ImageDataGenerator(
	rescale=1. / 255
	)

	test_generator = test_datagen.flow_from_directory(
    directory = args.dir,
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle = False
    # class_mode='binary'
    )

    # 将predict中得到的数据转化为混淆矩阵需要的参数
	predict = model.predict_generator(test_generator,verbose=1)
	print(predict)
	predict_y = np.argmax(predict,axis=1) # 将predict后的数组转化为实际预测的类（即预测结果）
	print(predict_y)
	true_y = test_generator.classes #验证集中true labe的标签
	print(true_y)

def predict(model, img, target_size):
	"""Run model prediction on image
	Args:
		model: keras model
		img: PIL format image
		target_size: (w,h) tuple
	Returns:
		list of predicted labels and their probabilities
	"""
	if img.size != target_size:
		img = img.resize(target_size)

	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	return preds[0]

# TODO
def batch_predict(model, img_dir, target_size): # a temp function
	"""Run model prediction on image
	Args:
		model: keras model
		img: PIL format image
		target_size: (w,h) tuple
	Returns:
		list of predicted labels and their probabilities
	"""
	if img.size != target_size:
		img = img.resize(target_size)

	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	return preds[0]

	# print("读取图片路径为：" + args.image_dir)
	# for root, dirs, files in os.walk(args.image_dir):
	# 	# print(dirs)
	# 	for file in files:
	# 		# print(file)
	# 		# plt.imshow(file)
	# 		# print(type(file))
	# 		# print(root + "/" + file)
	# 		file_name = root + "/" + file
	# 		img = Image.open(file_name)


	# 		pass

def plot_preds(image, preds):
	"""Displays image and the top-n predicted probabilities in a bar graph
	Args:
		image: PIL image
		preds: list of predicted labels and their probabilities
	"""
	plt.imshow(image)
	plt.axis('off')

	plt.figure()
	labels = ("cat", "dog")
	plt.barh([0, 1], preds, alpha=0.8)
	plt.yticks([0, 1], labels)
	plt.xlabel('Probability')
	# plt.xlabel('置信度分布')
	plt.xlim(0,1.01)
	plt.tight_layout()
	plt.show()


if __name__=="__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--image", help="path to image")
	a.add_argument("--image_url", help="url to image")
	a.add_argument("--image_dir", help="directory to image")
	a.add_argument("--dir", help="input testset path")
	a.add_argument("--model")
	args = a.parse_args()

	# if args.image is None and args.image_url is None and args.image_dir is None:
	if args.image is None and args.image_url is None and args.image_dir is None and args.dir is None:
		a.print_help()
		sys.exit(1)
		print("I am the error")

	model = load_model(args.model)

	if args.image is not None:
		img = Image.open(args.image)
		# print(img)
		preds = predict(model, img, target_size)
		print("---------------------------------start predit---------------------------------")
		plot_preds(img, preds)
		print(preds)
		print("the pic class is : "+ np.argmax(preds))

	if args.image_url is not None:
		response = requests.get(args.image_url)
		img = Image.open(BytesIO(response.content))
		preds = predict(model, img, target_size)
		plot_preds(img, preds)

	if args.dir is not None:
		data_predict()

	if args.image_dir is not None:
		# preds = batch_predict(model, args.image_dir, target_size)
		# print("读取图片路径为：" + args.image_dir)
		for root, dirs, files in os.walk(args.image_dir):
			# print(dirs)
			for file in files:
				# print(file)
				# plt.imshow(file)
				# print(type(file))
				# print(root + "/" + file)
				file_name = root + "/" + file
				img = Image.open(file_name)
				# all result
				print("---------------------------------start predict---------------------------------")
				# print("Now the " + file_name + " was predicted")
				# print("The result is ........")
				# start = time.time()
				preds = predict(model, img, target_size)
				# plot_preds(img, preds)
				print(preds)
				# print(time.time()-start)
				# print(type(preds))
				# print(type(preds[0]))

				# wrong result
				# float_preds = float(preds[0])
				# print(type(float_preds))
				# print(float_preds)
				# if float_preds < 0.5:
				# 	print("---------------------------------this file is wrong---------------------------------")
				# 	print(file_name + "\n")
				# 	print(preds)
					# plot_preds(img, preds)
				# print(type(root))
				# pass
		# plot_preds()