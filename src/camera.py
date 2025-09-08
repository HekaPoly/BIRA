from threading import Lock
import pyzed.sl as sl



class Camera:
    __lock = Lock() # i don<t know if it is better inside or outside the class
    __zed = None
    __init_params = None
        
    @staticmethod
    def get_camera(svo=None):
        with Camera.__lock:
            if Camera.__zed == None:
                Camera.__zed = sl.Camera()

                #if the order doesn't matter i will move this to the constructeur
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

                #should i make a open and close method


        return Camera.__zed
    
    def open():
        status = Camera.__zed.open(Camera.__init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

    def close():
        Camera.__zed.close()

    

    

    
