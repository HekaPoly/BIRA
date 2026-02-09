# revoir la documentation disponible sur OpenCV
# revoir les chiffres magiques et les adapter si nécessaire
import cv2
import numpy as np

class GlassDetection:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"Error: cannot load {image_path}")
    
    def is_depth_irreg(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # revoir le ratio, car il s'applique mieux aux petits objets
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
        middle_variance = np.var(middle_zone)
        is_uniform_top = top_variance < 1000
        
        return is_darker_at_bottom and (top_is_light or middle_is_light) and is_uniform_top
    
    def are_colors_irreg(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        edge_zone = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        has_weak_contours = len(contours) > 5
        
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
    print("Glass Detection")
    try:
        detectGlass = GlassDetection("PATH_TO_IMAGE.png")
        result = detectGlass.is_glass_obj_pres()
        print(f"Is a glass object present?\n{result}")
    except ValueError as e:
        print(f"Error: {e}")
