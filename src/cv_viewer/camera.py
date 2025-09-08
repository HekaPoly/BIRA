from threading import Lock
import pyzed.sl as sl



class Camera:
    __lock = Lock() # i don<t know if it is better inside or outside the class
    __zed = None
    __input_type = None
    __init_params = None
    __runtime_params = None
        
    @staticmethod
    def get_camera(svo=None):

        if Camera.__zed == None:
            with Camera.__lock:

                if Camera.__zed == None:
                    Camera.__zed = sl.Camera()

                    #if the order doesn<t matter i will move this to the constructeur
                    Camera.__input_type = sl.InputType()

                    if svo is not None:
                        Camera.__input_type.set_from_svo_file(svo) 

                    # Set the configuration parameters
                    Camera.__init_params = sl.InitParameters(input_t=Camera.__input_type, svo_real_time_mode=True)
                    Camera.__init_params.coordinate_units = sl.UNIT.METER
                    Camera.__init_params.depth_mode = sl.DEPTH_MODE.ULTRA
                    Camera.__init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
                    Camera.__init_params.depth_maximum_distance = 3
                    Camera.__init_params.camera_fps = 60

                    Camera.__runtime_params = sl.RuntimeParameters()

                    #should i make a open method
                    status = Camera.__zed.open(Camera.__init_params)
                    if status != sl.ERROR_CODE.SUCCESS:
                        print(repr(status))
                        exit()


        return Camera.__zed

    

    

    
