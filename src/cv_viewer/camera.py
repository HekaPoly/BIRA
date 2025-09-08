from threading import Lock, Thread
import pyzed.sl as sl



lock=Lock()

class Camera:

    def __init__(self, opt):
        self.__private_field="Private"
        self.opt = opt
        
    @staticmethod
    def get_private_field(self):

        if Camera.__private_field == None:
            lock.acquire()

            if Camera.__private_field == None:
                Camera.__private_field = sl.Camera
                input_type = sl.InputType()

                if self.opt.svo is not None:
                    input_type.set_from_svo_file(self.opt.svo) 

                init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
                init_params.coordinate_units = sl.UNIT.METER
                init_params.depth_mode = sl.DEPTH_MODE.ULTRA
                init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
                init_params.depth_maximum_distance = 3
                init_params.camera_fps = 60

                status = Camera.__private_field.open(init_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(repr(status))
                    exit()


        return Camera.__private_field
    

    

    
