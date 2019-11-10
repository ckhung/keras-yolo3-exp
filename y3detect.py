import argparse, os, re
import numpy as np
from keras.models import load_model
import cv2
from utils.bbox import BoundBox, bbox_iou, draw_boxes
from utils.utils import correct_yolo_boxes, do_nms, decode_netout, preprocess_input, labels

np.set_printoptions(threshold=np.nan)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to *.h5 weights file')

argparser.add_argument(
    '-o',
    '--out_dir',
    type=str, default='',
    help='directory of output images')

argparser.add_argument('image_paths', nargs='*',
    help='img1 img2 img3 ...')

def do_image_file(yolov3, image_path, out_dir=''):
    # set some parameters
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

    # preprocess the image
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    # run the prediction
    yolos = yolov3.predict(new_image)
    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     

    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh) 
 
    # write the image with bounding boxes to file
    if out_dir:
        out_path = out_dir + '/' + re.sub(r'.*/', '', image_path)
        print(out_path)
        cv2.imwrite(out_path, image.astype('uint8'))

if __name__ == '__main__':
    args = argparser.parse_args()
    yolov3 = load_model(args.weights)
    for img_file in args.image_paths:
        print(img_file + ' => ', end='')
        do_image_file(yolov3, img_file, out_dir=args.out_dir)
