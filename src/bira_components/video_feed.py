import os
import sys
import threading
import traceback
from time import perf_counter, sleep

import cv2
import numpy as np
import pyzed.sl as sl

from cv_viewer.tracking_viewer import render_2D
from cv_viewer.utils import generate_color_id_u


class _VisionRequest:
    def __init__(self):
        self.event = threading.Event()
        self.result = None


class VideoFeed:
    """
    Keeps the ZED camera open and shows a live video window with YOLO detections.

    A single background thread owns the camera and the vision models, so the ZED SDK and
    the GPU are never used from two threads at once. Live frames are detected without
    MobileSAM and without writing to the history log. When the state machine needs a real
    vision pass, request_detection() makes the thread run the full pipeline (with SAM
    refinement and history logging) on its next frame.

    The SAM overlay (toggle with 's') runs in its own worker thread, so the video keeps YOLO
    speed. Masks are matched to the current YOLO boxes by overlap, so they trail moving objects
    by the SAM latency (shown on screen as "age").

    If no display is available (e.g. SSH without X forwarding), the thread still runs and
    serves vision requests, it just doesn't open a window.
    """

    WINDOW_NAME = "BIRA vision"

    def __init__(self, camera, computer_vision, show_window=True, sam_overlay=False):
        self.__camera = camera
        self.__cv = computer_vision
        self.__show_window = show_window and self.__display_available()
        self.__thread = None
        self.__stop = threading.Event()
        self.__lock = threading.Lock()
        self.__pending = None
        self.__sam_overlay = sam_overlay
        self.__last_error = ""
        self.__fps = 0.0
        self.__last_tick = perf_counter()
        self.__draw_ms = 0.0
        # Async SAM: the loop submits the latest frame, the worker replaces __sam_result when done.
        self.__sam_cond = threading.Condition()
        self.__sam_job = None
        self.__sam_busy = False
        self.__sam_thread = None
        self.__sam_result = ([], [], 0.0, 0.0)  # boxes, masks, finished_at, duration_ms

    @staticmethod
    def __display_available():
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[VideoFeed] No display found (DISPLAY is not set). Video window disabled.")
            return False
        return True

    @property
    def running(self):
        return self.__thread is not None and self.__thread.is_alive()

    def start(self):
        if self.running:
            return
        # Opened here, in the caller's thread, so a camera failure is reported at startup.
        self.__camera.open()
        self.__stop.clear()
        self.__thread = threading.Thread(target=self.__run, name="VideoFeed", daemon=True)
        self.__thread.start()
        if self.__cv.segmenter is not None:
            self.__sam_thread = threading.Thread(target=self.__sam_worker, name="VideoFeedSAM", daemon=True)
            self.__sam_thread.start()
        print("[VideoFeed] Live video started. Press 's' in the video window to toggle SAM masks.")

    def request_detection(self, timeout=15.0):
        """
        Run the full vision pipeline on the next frame and wait for it.

        Returns:
            Tuple[sl.Objects, List[int]] | None: same as ComputerVision.detect_objects(),
            or None if the feed isn't running, the detection failed or it timed out.
        """
        if not self.running:
            return None
        request = _VisionRequest()
        with self.__lock:
            self.__pending = request
        if not request.event.wait(timeout):
            with self.__lock:
                if self.__pending is request:
                    self.__pending = None
            return None
        return request.result

    def __run(self):
        try:
            while not self.__stop.is_set():
                if self.__camera.grab() != sl.ERROR_CODE.SUCCESS:
                    sleep(0.01)
                    continue
                frame = self.__camera.get_frame()
                if frame is None:
                    continue

                with self.__lock:
                    request, self.__pending = self.__pending, None

                objects = None
                try:
                    if request is not None:
                        request.result = self.__cv.detect_objects(frame)
                        objects = request.result[0]
                    else:
                        objects = self.__cv.detect(frame, refine=False, log=False)
                        if self.__sam_overlay:
                            self.__submit_sam(frame)
                except Exception as exc:
                    self.__report_error("Detection", exc)
                finally:
                    if request is not None:
                        request.event.set()

                if self.__show_window:
                    self.__show(frame, objects)
        finally:
            if self.__show_window:
                try:
                    cv2.destroyWindow(self.WINDOW_NAME)
                except Exception:
                    pass
            self.__camera.close()

    def __sam_worker(self):
        while not self.__stop.is_set():
            with self.__sam_cond:
                while self.__sam_job is None and not self.__stop.is_set():
                    self.__sam_cond.wait(timeout=0.5)
                job, self.__sam_job = self.__sam_job, None
                if job is None:
                    continue
                self.__sam_busy = True
            try:
                image_rgb, boxes = job
                t = perf_counter()
                masks = self.__cv.segmenter.segment(image_rgb, boxes)
                self.__sam_result = (boxes, masks, perf_counter(), (perf_counter() - t) * 1000)
            except Exception as exc:
                self.__report_error("SAM", exc)
            finally:
                self.__sam_busy = False

    def __submit_sam(self, frame):
        """Hand the current frame to the SAM worker, unless it is still busy with the previous one."""
        boxes = [list(b[:4]) for b in self.__cv.last_yolo_boxes]
        if not boxes or self.__sam_busy or self.__cv.segmenter is None:
            return
        with self.__sam_cond:
            if self.__sam_job is None:
                self.__sam_job = (cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB), boxes)
                self.__sam_cond.notify()

    def __report_error(self, stage, exc):
        message = f"{stage} failed: {type(exc).__name__}: {exc}"
        if message != self.__last_error:  # print each distinct error once, not every frame
            print(f"[VideoFeed] {message}")
            traceback.print_exc()
        self.__last_error = message

    def __show(self, frame, objects):
        """Always imshow + waitKey, whatever happens while drawing, so the window and the keys stay alive."""
        now = perf_counter()
        self.__fps = 0.9 * self.__fps + 0.1 / max(now - self.__last_tick, 1e-6)
        self.__last_tick = now

        t_draw = perf_counter()
        display = frame.copy()  # the camera buffer is overwritten on the next grab
        n_objects = None
        n_masks = 0
        yolo_boxes = self.__cv.last_yolo_boxes
        if objects is not None:
            n_objects = len(objects.object_list)
            if self.__sam_overlay:
                try:
                    n_masks = self.__draw_masks(display, yolo_boxes)
                except Exception as exc:
                    self.__report_error("Mask drawing", exc)
            try:
                render_2D(display, [1.0, 1.0], objects, False, computer_vision=self.__cv)
            except Exception as exc:
                self.__report_error("Box drawing", exc)
            if n_objects == 0:
                # The ZED SDK returned nothing: draw YOLO's raw boxes (in red) so they are still visible.
                for x1, y1, x2, y2, cls_id, conf in yolo_boxes:
                    cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255, 255), 2)
                    cv2.putText(display, f"{self.__cv.get_label_name(cls_id)} {conf:.2f}", (int(x1), max(int(y1) - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)

        lines = [
            f"SAM: {'ON' if self.__sam_overlay else 'OFF'} (press s, window must be focused)",
            f"FPS: {self.__fps:.1f}  yolo: {len(yolo_boxes)}  zed objects: {'n/a' if n_objects is None else n_objects}  masks: {n_masks}",
        ]
        timings = self.__cv.timings
        _, _, sam_done, sam_ms = self.__sam_result
        sam_info = f"sam {sam_ms:.0f} (async, age {(perf_counter() - sam_done) * 1000:.0f})" if sam_done else "sam -"
        lines.append(f"ms: yolo {timings.get('yolo', 0):.0f}  zed {timings.get('zed', 0):.0f}  {sam_info}  draw {self.__draw_ms:.0f}")
        if self.__cv.zed_error:
            lines.append(self.__cv.zed_error[:90])
        if self.__last_error:
            lines.append(self.__last_error[:90])
        for i, line in enumerate(lines):
            cv2.putText(display, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 3)
            cv2.putText(display, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)

        self.__draw_ms = (perf_counter() - t_draw) * 1000
        try:
            cv2.imshow(self.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
        except Exception as exc:
            print(f"[VideoFeed] Video window failed ({exc}). Disabling it.")
            self.__show_window = False
            return
        if key == ord("s"):
            self.__sam_overlay = not self.__sam_overlay
            print(f"[VideoFeed] SAM overlay {'ON' if self.__sam_overlay else 'OFF'}")

    @staticmethod
    def __iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def __draw_masks(self, display, yolo_boxes):
        """Draw the latest async SAM masks on the current frame; returns how many were drawn.

        Each current YOLO box takes the mask of the most-overlapping box from the SAM frame.
        Only the box region is blended, not the whole frame.
        """
        sam_boxes, masks, _, _ = self.__sam_result
        height, width = display.shape[:2]
        drawn = 0
        for x1, y1, x2, y2, cls_id, _conf in yolo_boxes:
            best = max(range(len(sam_boxes)), key=lambda i: self.__iou((x1, y1, x2, y2), sam_boxes[i]), default=None)
            if best is None or self.__iou((x1, y1, x2, y2), sam_boxes[best]) < 0.5:
                continue
            mask = masks[best]
            if mask.shape != (height, width):
                continue
            # Region of the SAM frame's box, padded a little
            bx1, by1, bx2, by2 = sam_boxes[best]
            rx1, ry1 = max(int(bx1) - 5, 0), max(int(by1) - 5, 0)
            rx2, ry2 = min(int(bx2) + 5, width), min(int(by2) + 5, height)
            roi, roi_mask = display[ry1:ry2, rx1:rx2], mask[ry1:ry2, rx1:rx2]
            color = generate_color_id_u(cls_id)
            roi[roi_mask] = (roi[roi_mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, color, 2)
            drawn += 1
        return drawn

    def destroy(self):
        self.__stop.set()
        with self.__sam_cond:
            self.__sam_cond.notify_all()
        if self.__sam_thread is not None:
            self.__sam_thread.join(timeout=5)
            self.__sam_thread = None
        if self.__thread is not None:
            self.__thread.join(timeout=5)
            self.__thread = None
