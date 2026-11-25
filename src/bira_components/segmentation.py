import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


class Segmenter:
    """
    MobileSAM wrapper prompted with YOLO bounding boxes.

    Given an image and a list of boxes, returns one boolean mask per box. Masks are
    used by ComputerVision to compute a more robust 3D position than the box center
    (the median of the depth pixels that actually belong to the object).
    """

    # Number of erosion iterations applied to a mask before reading depth, to avoid
    # the unreliable depth values found on object borders.
    ERODE_ITERATIONS = 2
    # Minimum number of valid depth pixels needed to trust a mask-based position.
    MIN_VALID_PIXELS = 50

    @staticmethod
    def _default_weights():
        return str(Path(__file__).resolve().parents[2] / "models" / "mobile_sam.pt")

    def __init__(self, weights=None, device="cuda"):
        self.__weights = weights or self._default_weights()
        self.__device = device
        # The Ultralytics predictor is not thread-safe, and the live video worker and an on-demand
        # vision request can both call segment().
        self.__lock = threading.Lock()
        print("Initializing Network(MobileSAM)...")
        self.sam = SAM(self.__weights)
        print("MobileSAM initialized")

    def warmup(self):
        print("Warming up MobileSAM inference...")
        dummy_image = np.zeros((416, 416, 3), dtype=np.uint8)
        self.segment(dummy_image, [[100, 100, 300, 300]])
        print("MobileSAM inference ready.")

    def segment(self, image_rgb, boxes_xyxy):
        """
        Segment the object inside each box.

        Args:
            image_rgb (np.ndarray): HxWx3 RGB image
            boxes_xyxy (list[list[float]]): boxes as [x_min, y_min, x_max, y_max] in pixels

        Returns:
            list[np.ndarray]: one HxW boolean mask per box, same order as boxes_xyxy.
        """
        if len(boxes_xyxy) == 0:
            return []

        with self.__lock:
            results = self.sam.predict(
                image_rgb,
                bboxes=[list(map(float, box)) for box in boxes_xyxy],
                device=self.__device,
                save=False,
                verbose=False,
            )
        masks = results[0].masks
        if masks is None:
            return [np.zeros(image_rgb.shape[:2], dtype=bool) for _ in boxes_xyxy]
        return [m.astype(bool) for m in masks.data.cpu().numpy()]

    @classmethod
    def mask_position(cls, mask, xyz_map):
        """
        Compute the 3D position of a masked object from the ZED point cloud.

        Args:
            mask (np.ndarray): HxW boolean mask
            xyz_map (np.ndarray): HxWx(3|4) ZED MEASURE.XYZ array, in the camera frame

        Returns:
            np.ndarray | None: [x, y, z] median of the valid points under the mask, or None
            if too few valid depth pixels are available (caller should fall back).
        """
        if mask.shape != xyz_map.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), xyz_map.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=cls.ERODE_ITERATIONS).astype(bool)
        for candidate in (eroded, mask):
            points = xyz_map[candidate][:, :3]
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) >= cls.MIN_VALID_PIXELS:
                return np.median(points, axis=0)
        return None
