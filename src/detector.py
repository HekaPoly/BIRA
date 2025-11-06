#!/usr/bin/env python3

from ultralytics import YOLO
from threading import Lock, Thread

import numpy as np
import time
import cv2

import pyzed.sl as sl


import cv_viewer.tracking_viewer as cv_viewer
import cv_viewer.labels as lab
import history as rd
from utils import string_to_label

lock = Lock()
run_signal = False
exit_signal = False

MAX_DISTANCE: float = 7.0
PROXIMITY_THRESHOLD: float = 0.3

def xywh2abcd(xywh, im_shape):
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

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    yolo = YOLO(weights)
    yolo.model.to('cuda')
    yolo.model.eval()

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = yolo.predict(img, save=False, imgsz=img_size, conf=conf_thres,
                               iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        time.sleep(0.01)

def find_closest_object(new_position, object_dict, threshold):
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

def object_detection(duration: int, opt, label: int = -1) -> dict:

    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights,
                                                         'img_size': opt.img_size,
                                                         "conf_thres": opt.conf_thres})
    capture_thread.start()
    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create an InitParameters object and set the configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 3
    init_params.camera_fps = 60

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = False
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Utilities for 2D display
    image_left = sl.Mat()
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width,
                   display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                             [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                    init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4),
                               np.uint8)

    # Camera pose
    cam_w_pose = sl.Pose()
    
    # Set-up Timer
    timeout = time.time() + duration

    coordinate_dict = {}
    next_object_id = 0  # Counter for generating unique object IDs
    while not exit_signal:

        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                time.sleep(0.001)

            # Wait for detections
            lock.acquire()

            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()

            zed.retrieve_objects(objects, obj_runtime_param)

            object_list = objects.object_list
            for obj in object_list:
                if len(obj.bounding_box) == 0 : continue  
                if np.isnan(obj.position).any(): continue
                if obj.position[2] > MAX_DISTANCE: continue  # Filter outliers by distance.
                
                current_position = np.array(list(obj.position))
                
                # Retrieve or initialize the dictionary for the current label
                objects_dict = coordinate_dict.setdefault(obj.raw_label, {})
                
                # Find the closest object of the same label within the proximity threshold
                closest_id = find_closest_object(current_position, objects_dict, PROXIMITY_THRESHOLD)

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
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking, label)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            cv2.imshow("BIRA - Computer Vision", global_image)
            
            key = cv2.waitKey(10)
            current_time = time.time()
            if key == 27 or current_time > timeout:
                exit_signal = True

        else:
            exit_signal = True
    
    image_left.free()
    exit_signal = True
    zed.disable_object_detection()
    zed.close()

    return coordinate_dict

def exec_detection(label: str,  opt, duration: int=15):
    object_detection(duration, opt, lab.get_label_id(label))
