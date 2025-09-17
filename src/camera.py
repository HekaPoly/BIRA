from threading import Lock
import pyzed.sl as sl
import cv2



class Camera:
    __lock = Lock() # i don<t know if it is better inside or outside the class
    __zed = None
    __init_params = None

    def __init__(self, svo=None):
        self.__svo = svo
        
    @staticmethod
    def get_camera(svo=None):
        #WARNING svo is only used at the first call of this method
        with Camera.__lock:
            if Camera.__zed == None:
                Camera.__zed = sl.Camera()

                input_type = sl.InputType()

                if svo is not None:
                    input_type.set_from_svo_file(svo) 

                # Set the configuration parameters
                Camera.__init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
                Camera.__init_params.coordinate_units = sl.UNIT.METER
                Camera.__init_params.depth_mode = sl.DEPTH_MODE.ULTRA
                Camera.__init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
                Camera.__init_params.depth_maximum_distance = 3
                Camera.__init_params.camera_fps = 60

        return Camera.__zed
    
    def open(self):
        status = Camera.get_camera(self.__svo).open(Camera.__init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
            

    def close(self):
        if Camera.__zed is not None:
            Camera.__zed.close()
    
    @staticmethod
    def get_init_params():
        return Camera.__init_params

    def retrieveImage(self, image = sl.Mat(), view=sl.VIEW.LEFT, runtime_params=sl.RuntimeParameters()):
        if Camera.__zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            print("Image grabbed")
            with Camera.__lock:
                Camera.__zed.retrieve_image(image, view)
                return image.get_data()

    
if __name__ == "__main__":
    print("Camera test starting")
    cam = Camera()
    cam.open()
    
    while True:
        frame = cam.retrieveImage()
        if frame is not None:
            print("Image retrieved successfully.")
            cv2.imshow("Camera Frame left", cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
            
        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break
    
    cv2.destroyAllWindows()
    cam.close()