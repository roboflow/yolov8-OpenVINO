
import os
from datetime import datetime
import requests
import shutil

import torch

# Libraries for pre and post processsing
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
# from ultralytics.yolo.utils.plotting import Annotator, colors

# import onnx_runtime related package
import onnxruntime as rt

import numpy as np
import cv2


class ModelHandler(object):
    """
    A YOLOV8 Model handler implementation.
    """
    def __init__(self):
        
        self.initialized = False

        # Parameters for inference
        self.mlas_model = None
        self.ov_model = None
        self.input_names = None
        self.output_names = None

        # Parameters for pre-processing
        self.imgsz = 640 # default value for this usecase. 
        self.stride = 32 # default value for this usecase( differs based on the model selected )
        
        # Parameters for post-processing
        self.conf = 0.25
        self.iou = 0.45
        self.max_det = 300
        self.classes = None
        self.agnostic = False
        self.labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        
        self.path = '/home/raw-data/'

    def initialize(self, context):
        
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        so = rt.SessionOptions()
        self.mlas_model = rt.InferenceSession(os.path.join(model_dir,'yolov8n.onnx'), so, providers=['CPUExecutionProvider'])
        self.ov_model = rt.InferenceSession(os.path.join(model_dir,'yolov8n.onnx'), so, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type' : 'CPU_FP32'}]) 
        
        self.input_names = self.ov_model.get_inputs()[0].name
        outputs = self.ov_model.get_outputs()
        self.output_names = list(map(lambda output:output.name, outputs))
    
    def preprocess(self, request):
        
        if request and (',' in request[0]['body'].decode()):
            image_url = request[0]['body'].decode().split(',')[0].strip()
            device = request[0]['body'].decode().split(',')[1].strip()
        else:
            print("Inavalid input. Should be a comma seperated string with image url and device type.")
            return

        ## Set up the image URL and filename
        self.path = self.path+image_url.split("/")[-1] if self.path == '/home/raw-data/' else '/home/raw-data/'+image_url.split("/")[-1]
        # self.path = self.path+self.filename if self.path=='' else

        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True

            # Open a local file with wb ( write binary ) permission.
            with open(self.path,'wb') as f:
                shutil.copyfileobj(r.raw, f)

            print('Image sucessfully downloaded: ',self.path)
        else:
            print('Image couldn\'t be retreived')
            return
        
        image_abs_path = os.path.abspath(self.path)
        if os.path.isfile(image_abs_path) and image_abs_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:

            # Load Image
            img0 = cv2.imread(image_abs_path)

            # Padded resize
            img = LetterBox(self.imgsz, True, stride=self.stride)(image=img0.copy())

            # Convert
            img =  img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = img.astype(np.float32)  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
            if img.ndim == 3:
                img = np.expand_dims(img, axis=0)
            return img0, img, device
        else:
            print("Invalid image format.")
            return

    def inference(self, model_input, device):
        if device == 'cpu':
            print("Performing ONNX Runtime Inference with default CPU EP.")
            start_time = datetime.now()
            prediction = self.mlas_model.run(self.output_names, {self.input_names: model_input})
            end_time = datetime.now()
        elif device == 'CPU_FP32':
            print("Performing ONNX Runtime Inference with OpenVINO CPU EP.")
            start_time = datetime.now()
            prediction = self.ov_model.run(self.output_names, {self.input_names: model_input})
            end_time = datetime.now()
        else:
            print("Invalid Device Option. Supported device options are 'cpu', 'CPU_FP32'.")
            return None
        return prediction, (end_time - start_time).total_seconds()

    def postprocess(self, img0, img, inference_output):
        if inference_output is not None:
            prediction = inference_output[0]
            inference_time = inference_output[1]

            prediction = [torch.from_numpy(pred) for pred in prediction]
            preds = ops.non_max_suppression(prediction,
                                                    self.conf,
                                                    self.iou,
                                                    agnostic=self.agnostic,
                                                    max_det=self.max_det,
                                                    classes=self.classes)
            log_string = ''
            results = []
            for _, pred in enumerate(preds):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
                results.append(Results(img0, self.path, self.labels, boxes=pred))

            det = results[0].boxes
            if len(det) == 0:
                return log_string+'No detection found.'
            for c in det.cls.unique():
                n = (det.cls == c).sum()  # detections per class
                log_string += f"{n} {self.labels[int(c)]}{'s' * (n > 1)}, "

            raw_output = ''
            # annotator = Annotator(img0, pil=False)
            for d in reversed(det):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.labels[c]}' if d.id is not None else self.labels[c]
                # label = f'{name} {conf:.2f}'
                box = d.xyxy.squeeze().tolist()
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                raw_output+=f"name: {name}, confidence: {conf:.2f}, start_point: {p1}, end_point:{p2}\n"
                # annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

            # result_img = annotator.result()

            return [f"inference_time: {inference_time}s\nInference_summary: {log_string}\nraw_output:\n{raw_output}"]
        return None

    def handle(self, data, context):
        preprocessed_data = self.preprocess(data)
        if preprocessed_data:
            org_input, model_input, device = preprocessed_data
            inference_output = self.inference(model_input, device)
        return self.postprocess(org_input, model_input, inference_output)

_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    
    if data is None:
        return None

    return _service.handle(data, context)
