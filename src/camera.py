from threading import Lock
import pyzed.sl as sl
import cv2



class Camera:
    __lock = Lock()
    __instance =  None

    def __new__(cls, *args, **kwargs):
        if Camera.__instance is None:
            with Camera.__lock:
                if Camera.__instance is None:
                    Camera.__instance = super().__new__(cls)
                    Camera.__instance.isInitialised = False

        return Camera.__instance
    
    def __init__(self, svo=None):
        if self.isInitialised is False:
            with Camera.__lock:
                if self.isInitialised is False:
                    self.__zed = sl.Camera()
                    self.__image = sl.Mat()

                    input_type = sl.InputType()

                    if svo is not None:
                        input_type.set_from_svo_file(svo) 

                    # Set the configuration parameters
                    self.__init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
                    self.__init_params.coordinate_units = sl.UNIT.METER
                    self.__init_params.depth_mode = sl.DEPTH_MODE.ULTRA
                    self.__init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
                    self.__init_params.depth_maximum_distance = 3
                    self.__init_params.camera_fps = 60

                    self.isInitialised = True
            
    
    def get_camera(self):
        return self.__zed
    
    def get_init_params(self):
        return self.__init_params
    
    def open(self):
        if self.__zed.is_opened() is False:
            status = self.__zed.open(self.get_init_params())

            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()
            

    def close(self):
        self.__zed.close()
    

    def retrieve_image(self, view = sl.VIEW.LEFT, memory = sl.MEM.CPU, resolution = sl.Resolution(0, 0), runtime_params = sl.RuntimeParameters()):
        """Retrieves the next image from the camera
        Parameters:
            view (sl.VIEW): The image representation/view
            memory (sl.MEM): Place where the image will be stored
            resolution (sl.Resolution): The width and height of the image
            runtime_params (sl.RuntimeParameters): The parameters for sl.Camera.grab method
        Returns:
            np.ndarray or None: the array representing the image captured or None
        """
        if self.__zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            print("Image grabbed")
            with Camera.__lock:
                self.__zed.retrieve_image(self.__image, view, memory, resolution)
                return self.__image.get_data()
        
        return None

    def __del__(self):
        if hasattr(self, "_Camera__zed"):
            self.__zed.close()
        if hasattr(self, "_Camera__image"):
            self.__image.free()
    
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    

if __name__ == "__main__":
    print("Camera test starting")
    cam = Camera()
    cam2 = Camera()

    print("WOww the singleton works bro!" if cam is cam2 else "What the helly...")
    
    with cam:
        while True:
            frame = cam.retrieve_image()
            if frame is not None:
                print("Image retrieved successfully.")
                cv2.imshow("Camera Frame left", cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
                
            key = cv2.waitKey(1)
            if key == 27:  # Press 'ESC' to exit
                break
    
    cv2.destroyAllWindows()