import logging
import subprocess
import time
import os
from typing import Tuple
import cv2
from uuid import uuid4
import pandas as pd
from tqdm import tqdm
import re
import numpy as np

# ./Nobi_Trt \
#     --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
#     --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
#     --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show

# ./Nobi_Camera_AI_TensorRT \
#     --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
#     --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
#     --save-dir "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/nobi-hw-videocapture/EMoi///" \
#     --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show


class YoloV4_TensorRT():
    def __init__(self, exec_file: str, engine_file: str, names_file: str, dims: Tuple, obj_thres=0.5, nms_thres=0.45, type_yolo="csp"):
        # Use the path to the darknet file I sent you
        self.exec_file = exec_file
        self.engine_file = engine_file
        self.names_file = names_file
        self.width = dims[0]
        self.height = dims[1]
        self.obj_thres = obj_thres
        self.nms_thres = nms_thres
        self.type_yolo = type_yolo
        self.first_time = True
        self.pr = subprocess.Popen(
            [self.exec_file, "--engine-file", self.engine_file, "--label-file", self.names_file, "--dims", str(self.width), str(self.height),
             "--obj-thres", str(self.obj_thres), "--nms-thres", str(self.nms_thres), "--type-yolo", self.type_yolo, "--dont-show"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # , bufsize=1, universal_newlines=True)
        while True:
            output = self.pr.stdout.readline().strip().decode("utf-8")
            if "Enter Image Path:" in output:
                break

    def __str__(self):
        return " ".join([self.exec_file, "--engine-file", self.engine_file, "--label-file", self.names_file, "--dims", str(self.width), str(self.height),
                         "--obj-thres", str(self.obj_thres), "--nms-thres", str(self.nms_thres), "--type-yolo", self.type_yolo, "--dont-show"])

    def stop(self):
        self.pr.stdin.close()

    def process(self, camera_images):
        detected_objects = []
        if isinstance(camera_images, np.ndarray):
            camera_images = [camera_images]

        for cam_idx, camera_image in enumerate(camera_images):
            detections = []
            image_path = self.__preprocess(camera_image)
            self.pr.stdin.write((image_path + "\n").encode("utf-8"))
            self.pr.stdin.flush()
            while True:
                output = self.pr.stdout.readline().strip().decode("utf-8")
                if "Enter Image Path:" in output:
                    break
                label, confidient, x_left, y_top, width, height = re.findall(
                    "(.*):\s+(\d+)\%\s+x_left:\s+(.*)\s+y_top:\s+(.*)\s+width:\s+(.*)\s+height:\s+(.*)", output)[0]
                width = int(width)
                height = int(height)
                left_x = int(x_left)
                top_y = int(y_top)
                detections.append(
                    (label, confidient, (int(left_x), int(top_y), width, height), cam_idx))

            detected_objects = detected_objects + \
                self.__postprocess(camera_image, detections)
            os.remove(image_path)
        return detected_objects

    def __preprocess(self, camera_image):
        path = os.getcwd()
        image_path = path + "/" + uuid4().hex + ".jpg"
        cv2.imwrite(image_path, camera_image)
        return image_path

    def __postprocess(self, camera_image, detections):
        detected_objects = []
        for detection in detections:
            label, confidence, (x, y, w, h), cam_id = detection
            detected_objects.append([None,
                                     # camera_image,       # CameraImage
                                     label,              # str
                                     (x, y, w, h),       # BoundingBox
                                     confidence,         # int
                                     cam_id
                                     ])
        return detected_objects


class YoloV4_Darknet():
    def __init__(self, darknet: str, config_file: str, data_file: str, weights_file: str):
        # Use the path to the darknet file I sent you
        self.darknet = darknet
        self.config_file = config_file
        self.data_file = data_file
        self.weights_file = weights_file
        self.first_time = True

        assert os.path.exists(self.config_file)
        assert os.path.exists(self.weights_file)
        self.pr = subprocess.Popen(
            [self.darknet, 'detector', 'test', self.data_file, self.config_file, self.weights_file, '-dont_show', '-ext_output'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
        print(self.pr)

    def __str__(self):
        return " ".join([self.darknet, 'detector', 'test', self.data_file, self.config_file, self.weights_file, '-dont_show', '-ext_output'])

    def stop(self):
        self.pr.stdin.close()

    def process(self, camera_images, _ceil=True):
        detected_objects = []
        for cam_idx, camera_image in enumerate(camera_images):
            detections = []
            image_path = self.__preprocess(camera_image)
            output = ""
            self.pr.stdin.write(image_path + "\n")
            while True:
                c = self.pr.stdout.read(1)
                output += c
                if "Enter Image Path" in output:
                    if self.first_time:
                        output = ""
                        self.first_time = False
                        continue
                    break
            result = output.split("\n")
            for i in range(5, len(result)):
                data = result[i-1]
                label = data.split(":")[0]
                confidient = int(data.split(":")[1].split("%")[0])
                bbox = data.split("(")[1].split(")")[0]
                if _ceil:
                    left_x = int(re.search('left_x:(.*)top_y', bbox).group(1))
                    left_x = 0 if left_x < 0 else left_x
                    top_y = int(re.search('top_y:(.*)width', bbox).group(1))
                    top_y = 0 if top_y < 0 else top_y
                    width = int(re.search('width:(.*)height', bbox).group(1))
                    height = int(re.search('height:(.*)', bbox).group(1))
                else:
                    left_x = float(
                        re.search('left_x:(.*)top_y', bbox).group(1))
                    left_x = 0. if left_x < 0 else left_x
                    top_y = float(re.search('top_y:(.*)width', bbox).group(1))
                    top_y = 0. if top_y < 0 else top_y
                    width = float(re.search('width:(.*)height', bbox).group(1))
                    height = float(re.search('height:(.*)', bbox).group(1))

                detections.append(
                    (label, confidient, (left_x, top_y, width, height), cam_idx))

            detected_objects = detected_objects + \
                self.__postprocess(camera_image, detections)
            os.remove(image_path)
        return detected_objects

    def __preprocess(self, camera_image):
        path = os.getcwd()
        image_path = path + "/" + uuid4().hex + ".jpg"
        cv2.imwrite(image_path, camera_image)
        return image_path

    def __postprocess(self, camera_image, detections):
        detected_objects = []
        for detection in detections:
            label, confidence, (x, y, w, h), cam_id = detection
            detected_objects.append([
                camera_image,       # CameraImage
                label,              # str
                (x, y, w, h),       # BoundingBox
                confidence,         # int
                cam_id
            ])
        return detected_objects


class Nobi_Camera_AI_TensorRT():
    def __init__(self, exec_file: str, engine_file: str, names_file: str, save_dir: str, dims: Tuple, obj_thres=0.5, nms_thres=0.45, type_yolo="csp"):
        # Use the path to the darknet file I sent you
        self.exec_file = exec_file
        self.engine_file = engine_file
        self.names_file = names_file
        self.width = dims[0]
        self.height = dims[1]
        self.obj_thres = obj_thres
        self.nms_thres = nms_thres
        self.type_yolo = type_yolo
        self.save_dir = save_dir
        self.first_time = True
        self.pr = subprocess.Popen(
            [self.exec_file, "--engine-file", self.engine_file, "--label-file", self.names_file, "--save-dir", self.save_dir, "--dims", str(self.width), str(self.height),
             "--obj-thres", str(self.obj_thres), "--nms-thres", str(self.nms_thres), "--type-yolo", self.type_yolo, "--dont-show"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            output = self.pr.stdout.readline().strip().decode("utf-8")
            if "Enter COMMAND:" in output:
                break

    def __str__(self):
        return " ".join([self.exec_file, "--engine-file", self.engine_file, "--label-file", self.names_file, "--save-dir", self.save_dir, "--dims", str(self.width), str(self.height),
                         "--obj-thres", str(self.obj_thres), "--nms-thres", str(self.nms_thres), "--type-yolo", self.type_yolo, "--dont-show"])

    def __del__(self):
        self.stop()

    def stop(self):
        print("--STOP--")
        self.pr.stdin.write(("quit" + "\n").encode("utf-8"))
        self.pr.stdin.flush()
        self.pr.stdin.close()

    def process(self):
        detected_objects = []
        detections = [[], [], [], []]
        self.pr.stdin.write(("EMoi" + "\n").encode("utf-8"))
        self.pr.stdin.flush()
        while True:
            output = self.pr.stdout.readline().strip().decode("utf-8")
            if "Enter COMMAND:" in output:
                break

            re_fn = re.findall("\[FILENAME\]\s(.*)", output)
            re_cam = re.findall(
                "\[CAM\]\s(\d+)\s(.*)\s(\d+)%\sx_left:\s+(\d+)\s+y_top:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)", output)
            if re_fn:
                filename = re_fn[0]
            elif re_cam:
                cam_idx, label, confidient, x_left, y_top, width, height = re.findall(
                    "\[CAM\]\s(\d+)\s(.*)\s(\d+)%\sx_left:\s+(\d+)\s+y_top:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)", output)[0]
                cam_idx = int(cam_idx)
                width = int(width)
                height = int(height)
                x_left = int(x_left)
                y_top = int(y_top)
                detections[cam_idx].append(
                    (label, confidient, (int(x_left), int(y_top), width, height), cam_idx))

        camera_images = cv2.imread(filename)
        H, W, _ = camera_images.shape

        for dect, img in zip(detections, [camera_images[:H//2, :W//2, :], camera_images[:H//2, W//2:, :], camera_images[H//2:, :W//2, :], camera_images[H//2:, W//2:, :]]):
            detected_objects = detected_objects + self.__postprocess(img, dect)
        return detected_objects

    def __postprocess(self, camera_image, detections):
        detected_objects = []
        for detection in detections:
            label, confidence, (x, y, w, h), cam_id = detection
            detected_objects.append([camera_image,
                                     # camera_image,       # CameraImage
                                     label,              # str
                                     (x, y, w, h),       # BoundingBox
                                     confidence,         # int
                                     cam_id
                                     ])
        return detected_objects


class Nobi_Camera_AI_Darknet():
    def __init__(self, exec_file: str, weights_file: str, config_file: str, names_file: str, save_dir: str, obj_thres=0.5):
        # Use the path to the darknet file I sent you
        self.exec_file = exec_file
        self.weights_file = weights_file
        self.config_file = config_file
        self.names_file = names_file
        self.obj_thres = obj_thres
        self.save_dir = save_dir
        self.first_time = True
        self.pr = subprocess.Popen(
            [self.exec_file, "--weights-file", self.weights_file, "--cfg-file", self.config_file, "--names-file",
                self.names_file, "--save-dir", self.save_dir, "--thresh", str(self.obj_thres), "--dont-show"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(self)
        while True:
            output = self.pr.stdout.readline().strip().decode("utf-8")
            if "Enter COMMAND:" in output:
                break

    def __str__(self):
        return " ".join([self.exec_file, "--weights-file", self.weights_file, "--cfg-file", self.config_file, "--names-file", self.names_file, "--save-dir", self.save_dir, "--thresh", str(self.obj_thres), "--dont-show"])

    def __del__(self):
        self.stop()

    def stop(self):
        print("--STOP--")
        self.pr.stdin.write(("quit" + "\n").encode("utf-8"))
        self.pr.stdin.flush()
        self.pr.stdin.close()

    def process(self):
        detected_objects = []
        detections = [[], [], [], []]
        self.pr.stdin.write(("EMoi" + "\n").encode("utf-8"))
        self.pr.stdin.flush()
        while True:
            output = self.pr.stdout.readline().strip().decode("utf-8")
            if "Enter COMMAND:" in output:
                break

            re_fn = re.findall("\[FILENAME\] (.*)", output)
            re_cam = re.findall(
                "\[CAM\] (\d+)\t(.*)\t(\d+)%\tx_left:\s+(\d+)\s+y_top:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)", output)
            if re_fn:
                filename = re_fn[0]
            elif re_cam:
                cam_idx, label, confidient, x_left, y_top, width, height = re.findall(
                    "\[CAM\] (\d+)\t(.*)\t(\d+)%\tx_left:\s+(\d+)\s+y_top:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)", output)[0]
                cam_idx = int(cam_idx)
                width = int(width)
                height = int(height)
                x_left = int(x_left)
                y_top = int(y_top)
                detections[cam_idx].append(
                    (label, confidient, (int(x_left), int(y_top), width, height), cam_idx))
            else:
                pass
        camera_images = cv2.imread(filename)
        H, W, _ = camera_images.shape

        for dect, img in zip(detections, [camera_images[:H//2, :W//2, :], camera_images[:H//2, W//2:, :], camera_images[H//2:, :W//2, :], camera_images[H//2:, W//2:, :]]):
            detected_objects = detected_objects + self.__postprocess(img, dect)
        return detected_objects

    def __postprocess(self, camera_image, detections):
        detected_objects = []
        for detection in detections:
            label, confidence, (x, y, w, h), cam_id = detection
            detected_objects.append([camera_image,
                                     # camera_image,       # CameraImage
                                     label,              # str
                                     (x, y, w, h),       # BoundingBox
                                     confidence,         # int
                                     cam_id
                                     ])
        return detected_objects


def convert_box(detected_objects, cls_dict=None, type_output="xyxy"):
    return_boxes = []
    for d_o in detected_objects:
        _, label, (left_x, top_y, width, height), confidence, cam_id = d_o
        results = {}
        results.update(
            {"cls": cls_dict[label] if cls_dict is not None else label})

        results.update({"cfd": confidence})
        if type_output == "xyxy":
            results.update({"x1": int(left_x)})
            results.update({"y1": int(top_y)})
            results.update({"x2": int(left_x + width)})
            results.update({"y2": int(top_y + height)})
        elif type_output == "xywh":
            results.update({"x": int(left_x)})
            results.update({"y": int(top_y)})
            results.update({"w": int(width)})
            results.update({"h": int(height)})
        return_boxes.append(results)

    return return_boxes


if __name__ == "__main__":
    """
    yolov4_trt = YoloV4_TensorRT("./Nobi_Trt",
                                 "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine",
                                 "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names",
                                 (768, 768), 0.3, 0.3, "csp")
    """
    """
    yolov4_darknet = YoloV4_Darknet(darknet="./darknet",
                                    config_file="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.cfg",
                                    data_file="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.data",
                                    weights_file="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.weights")
    """
    nobi_camera_ai = Nobi_Camera_AI_TensorRT("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/build/Nobi_Camera_AI",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/EMoi",
                                             (512, 512), 0.5, 0.5, "csp")
    """
    nobi_camera_ai = Nobi_Camera_AI_Darknet("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/build/Nobi_Camera_AI",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.weights",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.cfg",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names",
                                             "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/EMoi", 0.5)
    """
    import ipdb; ipdb.set_trace()

    for i in tqdm(range(100)):
        m = nobi_camera_ai.process()
