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
    
    def __init__(self):
        if self.isInitialised is False:
            with Camera.__lock:
                if self.isInitialised is False:
                    self.__init_params = sl.InitParameters()
                    self.__zed = sl.Camera()
                    self.__camera_pose = sl.Pose()
                    self.__image = sl.Mat()
                    self.isInitialised = True
            
    
    def get_camera(self):
        return self.__zed
        
    def open(self, init_params = None):
        """Opens the zed camera.
        
        If it is not openned already. The program will end if the camera can't be openned.
        Parameters:
            init_params (sl.InitParameters): Class containing the options used to initialize the sl.Camera object.
        Returns: 
            None
        """
        if self.__zed.is_opened() is False:

            if init_params is None:
                status = self.__zed.open(self.__init_params)
            else:
                status = self.__zed.open(init_params)

            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()
    
    def grab(self, runtime_params = sl.RuntimeParameters()):
        """Grabs the latest images from the camera, rectify them, and compute the measurements based on the RuntimeParameters provided.
        Parameters:
            runtime_params (sl.RuntimeParameters): Contains parameters that defines the behavior of self.__zed.grab()
        Returns:
            ERROR_CODE: Describes if the grab was successfull or not
        """
        return self.__zed.grab(runtime_params)

    def get_frame(self, view = sl.VIEW.LEFT, memory = sl.MEM.CPU, resolution = sl.Resolution(0, 0)):
        """Retrieves images from the camera.
        Parameters:
            view (sl.VIEW): The image you want, left lens or right lens, rectified or unrectified and more
            memory (sl.MEM): The memory the image should be allocated
            resolution (sl.Resolution):  The Resolution you want for the image
        Returns:
            np.array or None: Returns a np.array representing the image if the retrieve was successfull, otherwise returns None
        """
        with Camera.__lock:
            if self.__zed.retrieve_image(self.__image, view, memory, resolution) == sl.ERROR_CODE.SUCCESS:
                return self.__image.get_data()
            else:
                return None
    
    def get_position(self):
        """Retrieves the estimated position and orientation of the camera with reference to the world frame.
        Parameters:
            None
        Returns:
            sl.Pose: Object containing positional tracking data giving the position and orientation of the camera in 3D space 
        """
        self.__zed.get_position(self.__camera_pose, sl.REFERENCE_FRAME.WORLD)
        return self.__camera_pose
    
    def __release(self):
        """Frees the memory allocated for the image
        Parameters:
        None
        Returns:
        None
        """
        self.__image.free()
        
    def save(self, filename, view = sl.VIEW.LEFT):
        """Saves the current image to a file.
        Parameters:
            filename (str): The path where the image will be saved
            view (sl.VIEW): The image you want to save, left lens or right lens, rectified or unrectified and more
        Returns:
            bool: True if the image was saved successfully, False otherwise
        """
        with Camera.__lock:
            return self.__zed.retrieve_image(self.__image, view) == sl.ERROR_CODE.SUCCESS and self.__image.write(filename) == sl.ERROR_CODE.SUCCESS
        

    
    def close(self):
        self.__zed.close()
        
    def __del__(self):
        if hasattr(self, "_Camera__zed"):
            self.__zed.close()
    
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__release()
        self.close()
    

if __name__ == "__main__":
    print("Camera test starting")
    cam = Camera()
    cam2 = Camera()

    cam.open()

    print("WOww the singleton works bro!" if cam is cam2 else "What the helly...")
    
    with cam:
        if (cam.grab()):
            frame = cam.get_frame()
            if frame is not None:
                print("Image retrieved successfully.")
                cam.save("camera_test_image.png")
                
        # while True:
        #     if (cam.grab()):
        #         frame = cam.get_frame()
        #         if frame is not None:
        #             print("Image retrieved successfully.")
        #             cv2.imshow("Camera Frame left", cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
                
        #     key = cv2.waitKey(1)
        #     if key == 27:  # Press 'ESC' to exit
        #         break
    
    if cam.get_frame() is None:
        print("Camera released successfully.")
    else :
        print("Camera release failed.")
    
    cv2.destroyAllWindows()