# YOLO3 (Detection, Training, and Evaluation)

Cloned from [experiencor/keras-yolo3](https://github.com/experiencor/keras-yolo3).
See original repo for full doc. Also recommended is the tutorial [How to Perform Object Detection With YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/).

All I did was rewriting yolo3_one_file_to_detect_them_all.py
into y3detect.py , making uses of utils/\*.py as much as possible.
The model conversion code is removed.
Use convert.py from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
to do the conversion just once.

```
wget https://pjreddie.com/media/files/yolov3.weights.
# 
python3 qqwweee-keras-yolo3/convert.py qqwweee-keras-yolo3/yolov3.cfg yolov3.weights yolo.h5
python3 y3detect.py -w yolo.h5 -o /tmp/result/ *.jpg
```

