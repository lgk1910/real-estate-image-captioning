import requests
import cv2
import os
import glob 
import numpy as np
import re
import detect
import urllib
from skimage import io
import glob
import os
import numpy as np
import requests
import random
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import concurrent.futures
import multiprocessing
import time

# physical_devices = tf.config.list_physical_devices('CPU')
# To find out which devices your operations and tensors are assigned to
tf.config.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)
# NOTE: uncomment this if train using GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

FURNITURE_TYPES = np.array(["tv", "conditioner", "sofa", "chair", "table", "view", "fan", "tv_shelf", "fridge"])
ROOM_TYPES = np.array(['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room'])
image_path = 'Resources/images/'
save_dir = 'runs/detect/exp/'
model_path = "Trained_model"
num_classes = 9

class Model:
	def __init__(self):
		# Navigate to yolov5 folder
		# %cd ~/../content/drive/MyDrive/Colab Notebooks/Furniture_detection/yolov5  
		self.room_model = load_model(model_path)
		# self.translator = Translator()
		pass

	def download_image_png(self, id, url, file_path):
		full_path = file_path + format(id, '04d') + '.png'
		try:
			img = io.imread(url)
			img=img[:,:,0:3]
			img_240 = cv2.resize(img, (240, 240,))/255.0
			img_416 = cv2.resize(img, (416, 416,))
# 			img_416 = cv2.cvtColor(img_416, cv2.COLOR_RGB2BGR)
			cv2.imwrite(full_path, img)
			return url, format(id, '04d') + '.png', img_240
		except:
			pass 
		return None

	def verbose_trans(self, str):
		if str == "view":
			return "open view"
		else:
			return str

	def load_txt(self, filename):
		output = np.zeros((1,num_classes))
		file = save_dir + filename + ".txt"
		with open(file, "r") as f:
			content = f.read()
			content = content[1:-1]
			z = re.split(r",", content)
			for ele in z:
				id = re.sub('[\.\s]', '', ele)
				try:
					output[0][int(id)] = 1
				except:
					# print('id:', id)
					pass
		return filename, FURNITURE_TYPES[output[0]==1].tolist()

	def detect_fur_and_transfer(self, filenames):
		detect.run()
		obj_list = {}
		with concurrent.futures.ThreadPoolExecutor() as executor:
			results = [executor.submit(self.load_txt, filename) for filename in filenames]
			for f in concurrent.futures.as_completed(results):
				result = f.result()
				obj_list[result[0]] = result[1]
		# return_dict['fur_detect'] = obj_list
		return obj_list

	def detect_room_and_transfer(self, images):
		classes = self.room_model.predict(images)
		pred_names = ROOM_TYPES[np.argmax(classes, axis = 1)]
		# return_dict['room_detect'] = pred_names
		return pred_names

	def predict(self, URLs):
		valid_urls = []
		filenames = []
		images = []
		start_time = time.monotonic()
		return_vals_lst = []
		try:
			# valid_urls, files = self.download_image_png(URLs, image_path)
			with concurrent.futures.ThreadPoolExecutor() as executor:
				results = [executor.submit(self.download_image_png, id, url, image_path) for id, url in enumerate(URLs)]

				for f in concurrent.futures.as_completed(results):
				# 	return_vals_lst.append(f.result())
					return_vals = f.result()
					if return_vals is not None:
						valid_urls.append(return_vals[0])
						filenames.append(return_vals[1])
						images.append(return_vals[2])
		except:
			pass
# 		for return_vals in return_vals_lst:
# 			if return_vals is not None:
# 				valid_urls.append(return_vals[0])
# 				filenames.append(return_vals[1])
# 				images.append(return_vals[2])
		# valid_urls = sorted(valid_urls, reverse=False)
		# filenames = sorted(filenames, reverse=False)
		# images = sorted(images, reverse=False)
		print(f"Download time: {time.monotonic() - start_time}")
		proc_start_time = time.monotonic()
		images = np.array(images)
		final_results = []
		results = []
		with concurrent.futures.ThreadPoolExecutor() as executor:
			results.append(executor.submit(self.detect_fur_and_transfer, filenames))
			results.append(executor.submit(self.detect_room_and_transfer, images))

			for f in concurrent.futures.as_completed(results):
				final_results.append(f.result())

		obj_list = final_results[0]
		pred_names = final_results[1]

		captions = []
		for filename, room_type in list(zip(filenames, pred_names)):
			furnitures = obj_list[filename]
			caption = ""
			choice = random.randint(0,1)
			if choice == 0:
				caption += f"{room_type} with "
			else:
				caption += f"{room_type} featuring "
			if len(furnitures) > 0:
				if len(furnitures) == 1:
					furniture = self.verbose_trans(furnitures[0])
					caption += f"{furniture}."
				else:
					for i in range(len(furnitures)):
						furniture = self.verbose_trans(furnitures[i])
						if i == len(furnitures) - 1:
							caption += f"and {furniture}."
						else:
							caption += f"{furniture}, "
			else:
				caption += "no furnitures."
			captions.append(caption)

		# translations = self.translator.translate(captions, dest='vi')
		# vietnamese_captions = [translation.text for translation in translations]
		# print(captions)
		# print(vietnamese_captions)
		print(f"Proc time: {time.monotonic() - proc_start_time}")
		# print(f"Time: {time.monotonic() - start_time}")
		return valid_urls, captions, captions
		# return valid_urls, captions, vietnamese_captions


	def delete_images(self, file_path):
		files = glob.glob(file_path + '*')
		count = len(files)
		for file in files:
			os.remove(file)
		print('{} files deleted'.format(count))