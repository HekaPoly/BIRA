import asyncio
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import argparse

from bira_components.camera import Camera
from bira_components import history as rd


class ComputerVision():

    @staticmethod
    def _default_opt():
        models_dir = Path(__file__).resolve().parents[2] / "models"
        return SimpleNamespace(
            weights=str(models_dir / "yolov8n.pt"),
            img_size=416,
            conf_thres=0.4,
        )

    def __init__(self, opt = None, camera=None):
        self.__detections = None
        default_opt = self._default_opt()
        if opt is None:
            self.__opt = default_opt
        else:
            self.__opt = SimpleNamespace(
                weights=getattr(opt, "weights", default_opt.weights),
                img_size=getattr(opt, "img_size", default_opt.img_size),
                conf_thres=getattr(opt, "conf_thres", default_opt.conf_thres),
            )
        self.__camera = camera

        print("Intializing Network(YOLO)...")
        self.yolo = YOLO(self.__opt.weights)
        self.yolo.model.to('cuda')
        self.yolo.model.eval()
        self._lock = asyncio.Lock()
        print("Network initialized")
        
    def detect_objects(self, frame):
        sl_object = self.detect(frame)
        detection_labels = [int(obj.label.item() if hasattr(obj.label, "item") else obj.label) for obj in self.__detections]
        return sl_object, detection_labels
        
        # TODO: remove old code after testing
        if message.keys().__contains__('detect_objects_1'):
            frame = message['detect_objects_1']
            if frame is None:
                return
            sl_object = self.detect(frame)
            # Send YOLO labels so SLM has detections even when ZED object_list is empty
            detection_labels = [int(obj.label.item() if hasattr(obj.label, "item") else obj.label) for obj in self.__detections]
            self.mediator.send_to(self, "SLM_Manager", {"detect_objects_ready": sl_object, "detection_labels": detection_labels})

    def __xywh2abcd(self, xywh):
        """Converts the bounding boxes from xywh format to abcd format
        Parameters:
            xywh (torch.Tensor): The bounding boxes in xywh format
        Returns:
            np.ndarray: The bounding boxes in abcd format
        """
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
        x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
        y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
        y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_min
        output[2][1] = y_max

        output[3][0] = x_max
        output[3][1] = y_max
        return output


    def __detections_to_custom_box(self, detections):
        """Converts externally detected objects into objects ingestable by ZED SDK
        Parameters:
            detections (YOLO.Boxes): The detection bounding boxes.
        Results:
            list[sl.CustomBoxObjectData]: Externally detected objects ingestable by ZED SDK 
        """
        output = []
        for det in detections:
            xywh = det.xywh[0]

            # Creating ingestable objects for the ZED SDK
            obj = sl.CustomBoxObjectData()
            obj.unique_object_id  = sl.generate_unique_id()
            obj.bounding_box_2d = self.__xywh2abcd(xywh)
            obj.label = det.cls
            obj.probability = det.conf
            obj.is_grounded = False
            output.append(obj)
        return output
            
    
    # def predict_async(self, audio_path):
    #     return asyncio.to_thread(
    #         self.yolo.predict,
    #         audio_path,
    #         imgsz=self.__opt.img_size,
    #         conf=self.__opt.conf_thres
    #     )
    
    def suppress_specular(img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Detect specular pixels: bright + low saturation
        specular_mask = (v > 220) & (s < 40)

        # Reduce intensity of specular highlights
        v = v.astype(np.float32)
        v[specular_mask] *= 0.6
        v = np.clip(v, 0, 255).astype(np.uint8)

        hsv_filtered = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
    
    def detect(self, frame, iou_thres=0.45):
        """Detects the objects present in the frame given.
        Parameters:
            frame (np.array): The image
            iou_thres (float): The intersection Over Union (IoU)
        Results:
            sl.Objects: The object containing the results of the detection.
        """
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        img = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        detections = self.yolo.predict(img, save=False, imgsz=self.__opt.img_size, conf=self.__opt.conf_thres,
                        iou=iou_thres)[0].cpu().numpy().boxes
        self.__detections = self.__detections_to_custom_box(detections)
        
        # -- Ingest detections
        self.__camera.get_camera().ingest_custom_box_objects(self.__detections)
        self.__camera.get_camera().retrieve_objects(objects, obj_runtime_param)
        
        object_list = objects.object_list
        rd.write_history(object_list)
        return objects
        
    
if __name__ == "__main__":
    models_dir = Path(__file__).resolve().parents[2] / "models"

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(models_dir / 'yolov8n.pt'), help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()

    cam = Camera()
    cam.open()

    cv = ComputerVision(opt=opt, camera=cam)

    with cam:
        while True:
            if (cam.grab()):
                frame = cam.get_frame()
                if frame is not None:
                    print(cv.detect(frame))
