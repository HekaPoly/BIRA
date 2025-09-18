import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
import cv2
import time
import cv_viewer.tracking_viewer as cv_viewer
import history as rd
import cv_viewer.labels as lab
from camera import Camera
import argparse



class ComputerVision:
    __lock = Lock()
    __MAX_DISTANCE: float = 7.0
    __PROXIMITY_THRESHOLD: float = 0.3

    def __init__(self, opt):
        self.__run_signal = False
        self.__exit_signal = False
        self.__image_net = None
        self.__detections = None
        self.__opt = opt
        self.__camera = Camera(opt.svo)
    
    #delete im_shape?
    def __xywh2abcd(self, xywh, im_shape):
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


    def __detections_to_custom_box(self, detections, im0):
        output = []
        for det in detections: #what is the purpose of i
            xywh = det.xywh[0]

            # Creating ingestable objects for the ZED SDK
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = self.__xywh2abcd(xywh, im0.shape)
            obj.label = det.cls
            obj.probability = det.conf
            obj.is_grounded = False
            output.append(obj)
        return output
    
    def __torch_thread(self, weights, img_size, conf_thres=0.2, iou_thres=0.45):
        print("Intializing Network...")

        yolo = YOLO(weights)
        yolo.model.to('cuda')
        yolo.model.eval()

        while not self.__exit_signal:
            if self.__run_signal:
                with ComputerVision.__lock:
                    img = cv2.cvtColor(self.__image_net, cv2.COLOR_BGRA2RGB)
                    # https://docs.ultralytics.com/modes/predict/#video-suffixes
                    det = yolo.predict(img, save=False, imgsz=img_size, conf=conf_thres,
                                    iou=iou_thres)[0].cpu().numpy().boxes

                    # ZED CustomBox format (with inverse letterboxing tf applied)
                    self.__detections = self.__detections_to_custom_box(det, self.__image_net)

                self.__run_signal = False
            time.sleep(0.01)
        
    def __find_closest_object(self, new_position, object_dict, threshold):
        """Find the id of closest existing object of the same label within threshold distance
        
        Find the ID of the closest existing object of the same label within a threshold distance.
        This function calculates the Euclidean distance between a given position (`new_position`) 
        and the last known position of each object in `object_dict`. It identifies the closest 
        object whose distance is less than or equal to the specified `threshold`.
        Parameters:
            new_position (numpy.ndarray): The position of the new object as a NumPy array.
            object_dict (dict): A dictionary where keys are object IDs and values are NumPy array of 
                positions associated with the object.
            threshold (float): The maximum distance within which an object is considered "close".
        Returns:
            int or None: The ID of the closest object if one is found within the threshold distance; 
                otherwise, returns None.
        """
        min_distance = float('inf')
        closest_obj_id = None

        for obj_id, positions in object_dict.items():
            if len(positions) > 0:
                last_position = positions[-1]
                distance = np.linalg.norm(new_position - last_position)
                if distance < min_distance and distance <= threshold:
                    min_distance = distance
                    closest_obj_id = obj_id
        
        return closest_obj_id
    
    def object_detection(self, duration: int, label: int = -1) -> dict:

        capture_thread = Thread(target=self.__torch_thread, kwargs={'weights': self.__opt.weights,
                                                            'img_size': self.__opt.img_size,
                                                            "conf_thres": self.__opt.conf_thres})
        capture_thread.start()

        print("Initializing Camera...")
        self.__camera.open()
        runtime_params = sl.RuntimeParameters()
        print("Initialized Camera")

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances
        # and boxes sticked to the ground.
        # positional_tracking_parameters.set_as_static = True
        Camera.get_camera().enable_positional_tracking(positional_tracking_parameters)

        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = False
        Camera.get_camera().enable_object_detection(obj_param)

        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

        # Display
        camera_infos = Camera.get_camera().get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution

        # Utilities for 2D display
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        image_scale = [display_resolution.width / camera_res.width,
                    display_resolution.height / camera_res.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                                [245, 239, 239, 255], np.uint8)

        # Utilities for tracks view
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                        Camera.get_init_params().depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4),
                                np.uint8)

        # Camera pose
        cam_w_pose = sl.Pose()
        
        # Set-up Timer
        timeout = time.time() + duration

        coordinate_dict = {}
        next_object_id = 0  # Counter for generating unique object IDs
        while not self.__exit_signal:

            if Camera.get_camera().grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

                # -- Get the image
                with ComputerVision.__lock:
                    self.__image_net = self.__camera.retrieve_image(sl.VIEW.LEFT)
                self.__run_signal = True

                # -- Detection running on the other thread
                while self.__run_signal:
                    time.sleep(0.001)

                # Wait for detections
                with ComputerVision.__lock:
                    # -- Ingest detections
                    Camera.get_camera().ingest_custom_box_objects(self.__detections)

                Camera.get_camera().retrieve_objects(objects, obj_runtime_param)

                object_list = objects.object_list
                for obj in object_list:
                    if len(obj.bounding_box) == 0 : continue  
                    if np.isnan(obj.position).any(): continue
                    if obj.position[2] > ComputerVision.__MAX_DISTANCE: continue  # Filter outliers by distance.
                    
                    current_position = np.array(list(obj.position))
                    
                    # Retrieve or initialize the dictionary for the current label
                    objects_dict = coordinate_dict.setdefault(obj.raw_label, {})
                    
                    # Find the closest object of the same label within the proximity threshold
                    closest_id = self.__find_closest_object(current_position, objects_dict, ComputerVision.__PROXIMITY_THRESHOLD)

                    if closest_id is not None:
                        # Append the position to the existing object's history
                        objects_dict[closest_id] = np.vstack([objects_dict[closest_id], current_position])
                    else:
                        # Create a new object with a unique ID and initialize its history
                        obj_id = next_object_id
                        next_object_id += 1
                        objects_dict[obj_id] = np.array([current_position])

                rd.write_history(object_list)
                
                # -- Display
                # Retrieve display data
                Camera.get_camera().get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                # 2D rendering
                np.copyto(image_left_ocv, 
                          self.__camera.retrieve_image(sl.VIEW.LEFT, sl.MEM.CPU, display_resolution))
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking, label)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                cv2.imshow("BIRA - Computer Vision", global_image)
                
                key = cv2.waitKey(10)
                current_time = time.time()
                if key == 27 or current_time > timeout:
                    self.__exit_signal = True
            else:
                self.__exit_signal = True
        
        self.__exit_signal = True
        Camera.get_camera().disable_object_detection()
        self.__camera.close()

        return coordinate_dict
    
    def exec_detection(self, label: str, duration: int=15):
        self.object_detection(duration, lab.get_label_id(label))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--cv', type=str, default=None, help='Showcase cv abilities of BIRA for specified duration (use inf for infinity)')
    parser.add_argument('--stt', action="store_true", help='Run speech to text app')
    parser.add_argument('--motors', help='Testing motors app')

    opt = parser.parse_args()

    cv = ComputerVision()
    cv.exec_detection("bouteille")



