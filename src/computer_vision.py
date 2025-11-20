import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import time
from camera import Camera
import argparse
import history as rd



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

        return np.array([
            [x_min, y_min],  # A
            [x_max, y_min],  # B
            [x_min, y_max],  # C
            [x_max, y_max]   # D
        ], dtype=float)



    def __detections_to_custom_box(self, detections):
        """Converts externally detected objects into objects ingestable by ZED SDK
        Parameters:
            detections (YOLO.Boxes): The detection bounding boxes.
        Results:
            list[sl.CustomBoxObjectData]: Externally detected objects ingestable by ZED SDK 
        """
        output = []
        for i, det in enumerate(detections):
            obj = sl.CustomBoxObjectData()
            xywh = det.xywh[0]

            # Creating ingestable objects for the ZED SDK
            obj.bounding_box_2d = self.__xywh2abcd(xywh)
            obj.label = int(det.cls.item())
            obj.probability = float(det.conf.item())
            
            # obj.is_3D = False
            obj.is_grounded = False
            
            output.append(obj)
        return output
            
    
    def detect(self, frame, iou_thres=0.4):
        """Detects the objects present in the frame given.
        Parameters:
            frame (np.array): The image
            iou_thres (float): The intersection Over Union (IoU)
        Results:
            sl.Objects: The object containing the results of the detection.
        """
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        yolo_detections = self.yolo.predict(
            img_rgb, 
            save=False, 
            imgsz=self.__opt.img_size, 
            conf=self.__opt.conf_thres,
            iou=iou_thres
        )[0]
        
        yolo_detections = yolo_detections.cpu().numpy().boxes if hasattr(yolo_detections, "boxes") else []
        
        zed_objects = self.__detections_to_custom_box(yolo_detections)
        

        # -- Ingest detections (Merge YOLO 2D boxes with the depth of the ZED)
        # with Camera.__lock:
        # Camera().get_camera().ingest_custom_box_objects(zed_objects)
        # Camera().get_camera().retrieve_objects(objects, obj_runtime_param)
        # print(f"Detected #2 {len(objects.object_list)} objects after ingestion")
        return zed_objects
        
    
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
    
    img = cv2.imread("/home/nvidia/Desktop/BRAs_VoiceAndVision/camera_test_image.png")
    img_array = np.array(img)
    # object_list = cv.detect(img_array).object_list
    zed_objects = cv.detect(img_array)
    
    print(f"Detected {len(zed_objects)} objects")
    for obj in zed_objects:
        print(f"Label: {obj.label}, Probability: {obj.probability}, BBox 2D: {obj.bounding_box_2d}\n")



    # with cam:
    #     while True:
    #         if (cam.grab()):
    #             frame = cam.get_frame()
    #             if frame is not None:
    #                 object_list =cv.detect(frame).object_list
    #                 print(cv.detect(frame))
    #                 print(object_list)
    #                 rd.write_history(object_list)
                    
                    
    
