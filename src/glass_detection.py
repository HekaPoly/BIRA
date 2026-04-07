from camera import Camera
import pyzed.sl as sl
import cv2
import numpy as np

def get_depth_from_camera(cam):
    zed = cam.get_camera()
    depth_mat = sl.Mat()
    if zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH) == sl.ERROR_CODE.SUCCESS:
        return depth_mat.get_data().copy()
    return None

class GlassDetection:
    def __init__(self, image, depth_map=None):
        if isinstance(image, str):
            self.image = cv2.imread(image)
            if self.image is None:
                raise ValueError(f"Error: cannot load {image}")
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise ValueError("Image must be a file path or a numpy array")
        self.depth_map = depth_map

    # TODO: Revoir les chiffres magiques et les adapter
    def is_depth_irreg(self):
        if self.depth_map is not None:
            return self._depth_map_irreg()
        return self._brightness_fallback()

    def _depth_map_irreg(self):
        depth = self.depth_map

        invalid_mask = ~np.isfinite(depth)
        invalid_ratio = np.sum(invalid_mask) / depth.size
        has_invalid_pixels = 0.03 < invalid_ratio < 0.60

        valid_depth = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
        depth_grad = cv2.Laplacian(valid_depth, cv2.CV_64F)
        total_valid = np.sum(~invalid_mask)
        sharp_disc = np.sum(np.abs(depth_grad) > 1.0)
        has_discontinuities = (sharp_disc / max(total_valid, 1)) > 0.05

        return has_invalid_pixels and has_discontinuities

    def _brightness_fallback(self):
        # TODO: Revoir le ratio (car il s'applique mieux aux petits objets)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        third_height = height // 3

        top_zone = gray[0:third_height, :]
        middle_zone = gray[third_height:2*third_height, :]
        bottom_zone = gray[2*third_height:, :]

        top_brightness = np.mean(top_zone)
        middle_brightness = np.mean(middle_zone)
        bottom_brightness = np.mean(bottom_zone)

        is_darker_at_bottom = bottom_brightness < top_brightness - 20
        is_darker_at_bottom = is_darker_at_bottom and (bottom_brightness < middle_brightness - 15)

        top_is_light = top_brightness > 150
        middle_is_light = middle_brightness > 140

        top_variance = np.var(top_zone)
        # middle_variance = np.var(middle_zone)
        # bottom_variance = np.var(bottom_zone)
        is_uniform_top = top_variance < 1000

        return is_darker_at_bottom and (top_is_light or middle_is_light) and is_uniform_top

    def are_colors_irreg(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edge_zone = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        has_weak_contours = len(contours) < 5

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_score < 100

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_blue, mask_green)
        combined_mask = cv2.bitwise_or(combined_mask, mask_white)

        colors_at_edges = cv2.bitwise_and(combined_mask, edge_zone)
        edge_pixels = np.sum(edge_zone > 0)

        if edge_pixels > 0:
            colored_edge_pixels = np.sum(colors_at_edges > 0)
            edge_ratio = colored_edge_pixels / edge_pixels
            has_edge_colors = edge_ratio > 0.2
        else:
            has_edge_colors = False

        reflection_pixels = np.sum(mask_white > 0)
        total_pixels = self.image.shape[0] * self.image.shape[1]
        reflection_ratio = reflection_pixels / total_pixels
        has_reflections = reflection_ratio > 0.05

        colored_pixels = np.sum(combined_mask > 0)
        color_ratio = colored_pixels / total_pixels
        has_sparse_colors = color_ratio < 0.3

        color_presence = has_edge_colors or has_reflections
        texture_criteria = is_blurry or has_weak_contours

        return color_presence and texture_criteria and has_sparse_colors

    def is_glass_obj_pres(self):
        depth_detected = self.is_depth_irreg()
        colors_detected = self.are_colors_irreg()

        return colors_detected and depth_detected

if __name__ == "__main__":
    print("========== Glass Detection ==========")

    print("Camera started")
    cam = Camera()
    # print("TEST1")
    cam.open()
    # print("TEST2")

    depth_warning_shown = False

    with cam:
        while True:
            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                frame = cam.get_frame()
                depth = get_depth_from_camera(cam)
                if depth is None and not depth_warning_shown:
                    print("Warning: depth map is unavailable, using brightness fallback.")
                    depth_warning_shown = True
                if frame is not None:
                    cam_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    glassDetection = GlassDetection(cam_frame_bgr, depth)
                    result = glassDetection.is_glass_obj_pres()
                    print(f"Is a glass object present?\n{result}")
                    print("Image retrieved successfully.")
                    cv2.imshow("Camera Frame", cam_frame_bgr)

            key = cv2.waitKey(1)
            if key == 27:  # Press 'ESC' to exit
                break

    cv2.destroyAllWindows()
