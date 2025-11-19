from computer_vision import ComputerVision, Camera, np, sl, cv2, argparse, time
import cv_viewer.tracking_viewer as cv_viewer

if __name__ == "__main__":
    #setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()

    myComputerVision = ComputerVision(opt)
    myCamera = Camera()
    myCamera.open()

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = False
    myCamera.get_camera().enable_object_detection(obj_param)

    cameraInfo = myCamera.get_camera().get_camera_information()
    camera_res = cameraInfo.camera_configuration.resolution
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    
    image_scale = [display_resolution.width / camera_res.width,
                   display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                             [245, 239, 239, 255], np.uint8)
    
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4),
                               np.uint8)
    
    #display
    with myCamera:
        while True:
            if (myCamera.grab()):
                frame = myCamera.get_frame()
                if frame is not None:
                    myCamera.get_position()

                    # 2D rendering
                    np.copyto(image_left_ocv, frame)
                    cv_viewer.render_2D(image_left_ocv, image_scale, myComputerVision.detect(frame), obj_param.enable_tracking, -1)
                    global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                    cv2.imshow("BIRA - Computer Vision", global_image)

                    key = cv2.waitKey(10)
                    current_time = time.time()
                    if key == 27:
                        break