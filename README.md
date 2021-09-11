# Setup environment
## Upgrade pip & install packages
```
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio
cat requirements.txt | xargs -n 1 pip3 install
```
## Install tensorflow gpu
Step 1: Visit https://www.tensorflow.org/install/pip#package-location and choose the tensorflow GPU version that is compatible with your python version

Step 2: Run `python -m pip install <wheel_url>` with `<wheel_url>` is the URL you've just got from step 1.

Step 3: Wait for the installation and finish.

# Download pretrained models
## Download room_type detection model
Room_type detection model available [here](https://drive.google.com/drive/folders/1RW6bJIPujzdMnum849keACUcqpk-7DHk?usp=sharing). Download the Trained_model folder and put it in the root of the repository.
## Download pretrained YOLOv5 furniture detection model
YOLOv5 pretrained model availabel [here](https://drive.google.com/drive/folders/10m6lruhTZRUfyhTuViICfW1O1-qR8AE-?usp=sharing). Download the yolov5s_results4 folder and put it in the runs/train directory.
# Host API
```
python3 app.py
```

# Send request
```
python3 request.py
```

# Output
The ouput will be stored in the file `res.json`, whose content has the same structure as below:

![image](https://user-images.githubusercontent.com/57819211/124773390-1e6cbb80-df67-11eb-8e51-894fdaa83c7c.png)

**Note**: It can be seen that there are two captions generated for each URL. The second caption is just for demonstration purpose since it's supposed to be in  Vietnamese. However, due to the fact that we don't have a subscribed account on Google Cloud Platform, we can't use Google Translate API to translate the captions to Vietnamese.
