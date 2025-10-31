import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import time
from camera import Camera
import argparse



class ComputerVision:

    def __init__(self, opt = None):
        self.__detections = None
        self.__opt = opt

        print("Intializing Network(YOLO)...")
        self.yolo = YOLO(opt.weights)
        self.yolo.model.to('cuda')
        self.yolo.model.eval()
        print("Network initialized")
    
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
            obj.bounding_box_2d = self.__xywh2abcd(xywh)
            obj.label = det.cls
            obj.probability = det.conf
            obj.is_grounded = False
            output.append(obj)
        return output
            
    
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
        Camera().get_camera().ingest_custom_box_objects(self.__detections)
        Camera().get_camera().retrieve_objects(objects, obj_runtime_param)
        return objects
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()

    cv = ComputerVision(opt)

    cam = Camera()
    cam.open()

    with cam:
        while True:
            if (cam.grab()):
                frame = cam.get_frame()
                if frame is not None:
                    print(cv.detect(frame))
