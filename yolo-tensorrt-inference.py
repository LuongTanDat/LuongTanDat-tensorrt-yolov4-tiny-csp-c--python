import sys
import os
from time import time
import math
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import itertools
from functools import partial
import struct  # get_image_size
import imghdr  # get_image_size
import multiprocessing

engine_path = "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4/yolov4-608.engine"
image_path = "/mnt/E83E38E23E38AB86/Users/kikai/Desktop/yolov4-tensorrt/samples/bus.jpg"
namesfile = "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4/yolov4-608.names"

def sigmoid(inp, model_name=""):
    if model_name == "csp":
        return inp
    else:
        return 


class Yolov4():
    class DectectRes():
        def __init__(self, **kwargs):
            self.classes = kwargs["classes"]
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.w = kwargs["w"]
            self.h = kwargs["h"]
            self.prob = kwargs["prob"]

        def __str__(self):
            return f"classes: {self.classes}\tx: {self.x}\ty: {self.y}\tw: {self.w}\th: {self.h}\tprob: {self.prob}"

    class Point2D():
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __str__(self):
            return "({0},{1})".format(self.x, self.y)

        def __add__(self, other):
            x = self.x + other.x
            y = self.y + other.y
            return Yolov4.Point2D(x, y)

        def __sub__(self, other):
            x = self.x - other.x
            y = self.y - other.y
            return Yolov4.Point2D(x, y)

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, **kwargs):
        self.engine_file = kwargs["engine_file"]
        self.labels_file = kwargs["labels_file"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.INPUT_CHANNEL = kwargs["INPUT_CHANNEL"]
        self.IMAGE_WIDTH = kwargs["IMAGE_WIDTH"]
        self.IMAGE_HEIGHT = kwargs["IMAGE_HEIGHT"]
        self.obj_threshold = kwargs["obj_threshold"]
        self.nms_threshold = kwargs["nms_threshold"]
        self.model_name = kwargs["model"]
        self.strides = kwargs["strides"]
        self.num_anchors = kwargs["num_anchors"]
        self.anchors = kwargs["anchors"]
        self.refer_matrix = []
        self.TRT_LOGGER = trt.Logger()

        # buffer
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        self.detect_labels = self.readCOCOLabel(self.labels_file)
        self.CATEGORY = len(self.detect_labels)
        self.grids = np.array([[self.num_anchors[index], int(self.IMAGE_HEIGHT/stride), int(
            self.IMAGE_WIDTH/stride)] for index, stride in enumerate(self.strides)])

        self.refer_rows = sum([self.num_anchors[index] * int(self.IMAGE_HEIGHT/stride) * int(
            self.IMAGE_WIDTH/stride) for index, stride in enumerate(self.strides)])
        self.refer_cols = 6
        self.GenerateReferMatrix()
        self.pool = multiprocessing.Pool(processes=6)
        self.sigmoid = (lambda i : i) if self.model_name == "csp" else (lambda inp: 1. / (1.0 + math.exp(-inp)))

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def readCOCOLabel(self, namesfile):
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def LoadEngine(self):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(self.engine_file))
        with open(self.engine_file, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        for binding in self.engine:

            size = trt.volume(self.engine.get_binding_shape(
                binding)) * self.BATCH_SIZE
            dims = self.engine.get_binding_shape(binding)

            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            self.host_mem = cuda.pagelocked_empty(size, dtype)
            self.device_mem = cuda.mem_alloc(self.host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(self.device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(Yolov4.HostDeviceMem(
                    self.host_mem, self.device_mem))
            else:
                self.outputs.append(Yolov4.HostDeviceMem(
                    self.host_mem, self.device_mem))

        # return inputs, outputs, bindings, stream

    def GenerateReferMatrix(self):
        for n in range(self.grids.shape[0]):
            for c in range(self.grids[n][0]):
                anchor = self.anchors[n * self.grids[n][0] + c]
                for h in range(self.grids[n][1]):
                    for w in range(self.grids[n][2]):
                        self.refer_matrix.append(
                            [w, self.grids[n][2], h, self.grids[n][1], anchor[0], anchor[1]])

        self.refer_matrix = np.array(self.refer_matrix)

    def EngineInference(self, img):
        HEIGHT, WIDTH = img.shape[:2]
        start = time()
        img_in = self.prepareImage(img)
        # print("== prepareImage ==:\t", time() - start)
        # inputs, outputs, bindings, stream = buffers
        start = time()
        self.inputs[0].host = img_in
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
         for inp in self.inputs]
        # Run inference.
        self.context.execute_async(
            bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
         for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        trt_outputs = [out.host for out in self.outputs]
        # print("== Inference ==:\t", time() - start)
        import ipdb; ipdb.set_trace()
        start = time()
        result = self.postProcess(trt_outputs[0], WIDTH, HEIGHT)
        print("== postProcess ==:\t", time() - start)
        return result

    def prepareImage(self, img):
        # Input
        resized = cv2.resize(
            img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        return img_in

    def postProcess(self, outputs, WIDTH, HEIGHT):
        # import ipdb; ipdb.set_trace()
        ratio_w = float(WIDTH) / self.IMAGE_WIDTH
        ratio_h = float(HEIGHT) / self.IMAGE_HEIGHT
        outputs = np.reshape(outputs, (-1, self.CATEGORY + 5))
        result = []
        for idx, row in enumerate(outputs):
            start = time()
            cls = np.argmax(row[5:])
            prob = self.sigmoid(row[4]) * self.sigmoid(row[cls + 5])
            if prob < self.obj_threshold:
                continue

            start = time()
            anchor = self.refer_matrix[idx]
            x = (self.sigmoid(row[0]) + anchor[0]) / anchor[1] * float(WIDTH)
            y = (self.sigmoid(row[1]) + anchor[2]) / anchor[3] * float(HEIGHT)
            start = time()
            if self.model_name == "csp":
                w = float(pow(row[2] * 2, 2)) * anchor[4] * ratio_w
                h = float(pow(row[3] * 2, 2)) * anchor[5] * ratio_h
            else:
                w = math.exp(row[2]) * anchor[4] / \
                    float(self.IMAGE_WIDTH) * float(WIDTH)
                h = math.exp(row[3]) * anchor[5] / \
                    float(self.IMAGE_HEIGHT) * float(HEIGHT)
            result.append(Yolov4.DectectRes(
                classes=cls, x=x, y=y, w=w, h=h, prob=prob))

        result = self.NmsDetect(result)
        return result

    def _sigmoid(self, inp):
        if self.model_name == "csp":
            return inp
        else:
            # return 1. / (1.0 + math.exp(-inp))
            return logistic.cdf(inp)

    def NmsDetect(self, detections):
        detections = sorted(detections, key=lambda d: d.prob, reverse=True)

        for i in range(len(detections)):
            for j in range(i+1, len(detections)):
                if detections[i].classes == detections[j].classes:
                    iou = self.IOUCalculate(detections[i], detections[j])
                    if (iou > self.nms_threshold):
                        detections[j].prob = 0

        for i in range(len(detections)-1, -1, -1):
            if detections[i].prob == 0:
                detections.remove(detections[i])

        return detections

    def IOUCalculate(self, det_a, det_b):
        center_a = Yolov4.Point2D(det_a.x, det_a.y)
        center_b = Yolov4.Point2D(det_b.x, det_b.y)
        left_up = Yolov4.Point2D(min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                                 min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2))
        right_down = Yolov4.Point2D(max(det_a.x + det_a.w / 2, det_b.x +
                                    det_b.w / 2), max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2))

        distance_d = (center_a - center_b).x * (center_a - center_b).x + \
            (center_a - center_b).y * (center_a - center_b).y
        distance_c = (left_up - right_down).x * (left_up - right_down).x + \
            (left_up - right_down).y * (left_up - right_down).y
        inter_l = det_a.x - det_a.w / \
            2 if (det_a.x - det_a.w / 2 > det_b.x -
                  det_b.w / 2) else det_b.x - det_b.w / 2
        inter_t = det_a.y - det_a.h / \
            2 if (det_a.y - det_a.h / 2 > det_b.y -
                  det_b.h / 2) else det_b.y - det_b.h / 2
        inter_r = det_a.x + det_a.w / \
            2 if (det_a.x + det_a.w / 2 < det_b.x +
                  det_b.w / 2) else det_b.x + det_b.w / 2
        inter_b = det_a.y + det_a.h / \
            2 if (det_a.y + det_a.h / 2 < det_b.y +
                  det_b.h / 2) else det_b.y + det_b.h / 2
        if (inter_b < inter_t) or (inter_r < inter_l):
            return 0
        inter_area = (inter_b - inter_t) * (inter_r - inter_l)
        union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area
        if union_area == 0:
            return 0
        else:
            return inter_area / union_area - distance_d / distance_c


if __name__ == '__main__':
    # "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4/yolov4-608.engine";
    # "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4/yolov4-608.names";
    # "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4-csp/yolov4-csp-512.engine";
    # "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4-csp/yolov4-csp-512.names";

    yolo = Yolov4(engine_file="/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4/yolov4-608.engine",
                  labels_file="/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4/yolov4-608.names",
                  BATCH_SIZE=1,
                  INPUT_CHANNEL=3,
                  IMAGE_WIDTH=608,
                  #   IMAGE_WIDTH=512,
                  IMAGE_HEIGHT=608,
                  #   IMAGE_HEIGHT=512,
                  obj_threshold=0.5,
                  #   obj_threshold=0.6,
                  nms_threshold=0.45,
                  model="",
                  #   model="csp",
                  strides=(8, 16, 32),
                  num_anchors=(3, 3, 3),
                  anchors=((12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)))

    yolo.LoadEngine()
    image = cv2.imread(
        "/mnt/E83E38E23E38AB86/Users/kikai/Desktop/yolov4-tensorrt/samples/bus.jpg")
    from tqdm import tqdm
    # for _ in tqdm(range(1000)):
    #     result = yolo.EngineInference(image)

    result = yolo.EngineInference(image)

