import requests
import cv2
import os
import glob 
import numpy as np
import re
import detect

image_shape = 416
image_path = 'Resources/images/'
save_dir = 'runs/detect/exp/'
num_classes = 9

class Model:
  def __init__(self):
    # Navigate to yolov5 folder
    # %cd ~/../content/drive/MyDrive/Colab Notebooks/Furniture_detection/yolov5  
    pass

  def download_image_ipg(self, urls, file_path):
    for i, url in enumerate(urls):
      full_path = file_path + format(i, '04d') + '.png'
      r = requests.get(url)
      with open(full_path, 'wb') as outfile:
          outfile.write(r.content)
      img = cv2.imread(full_path)
      img = cv2.resize(img, (416,416,))
      cv2.imwrite(full_path, img)

  def predict(self, URLs):
    try:
      self.download_image_ipg(URLs, image_path)
    except:
      pass

    # os.system('python detect.py --weights runs/train/yolov5s_results4/weights/best.pt --img 416 --conf 0.4 --source {}'.format(image_path))
    detect.run()

  def delete_images(self, file_path):
    files = glob.glob(file_path + '*')
    count = len(files)
    for file in files:
      os.remove(file)
    print('{} files deleted'.format(count))

  def transform_vec(self):
    for file in glob.glob(save_dir + "*.txt"):
      # print(file)
      output = np.zeros((1,num_classes))
      with open(file, "r") as f:
        content = f.read()
      #   print(content)
        content = content[1:-1]
        z = re.split(r",", content)
        for ele in z:
          id = re.sub('[\.\s]', '', ele)
          # print(int(id))
          output[0][int(id)] = 1
        # print(output[0])
      with open(file, "w") as f:
        f.write(str(output[0]))
        print(file + ' written successfully')
  