import os
import sys
import threading
from time import sleep

import cv2
import pyzed.sl as sl

from cv_viewer.tracking_viewer import render_2D


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

    If no display is available (e.g. SSH without X forwarding), the thread still runs and
    serves vision requests, it just doesn't open a window.
    """

    WINDOW_NAME = "BIRA vision"

    def __init__(self, camera, computer_vision, show_window=True):
        self.__camera = camera
        self.__cv = computer_vision
        self.__show_window = show_window and self.__display_available()
        self.__thread = None
        self.__stop = threading.Event()
        self.__lock = threading.Lock()
        self.__pending = None

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
        print("[VideoFeed] Live video started.")

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
                except Exception as exc:
                    print(f"[VideoFeed] Detection failed: {exc}")
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

    def __show(self, frame, objects):
        try:
            display = frame.copy()  # the camera buffer is overwritten on the next grab
            if objects is not None:
                render_2D(display, [1.0, 1.0], objects, False, computer_vision=self.__cv)
            cv2.imshow(self.WINDOW_NAME, display)
            cv2.waitKey(1)  # required for the window to refresh
        except Exception as exc:
            print(f"[VideoFeed] Video window failed ({exc}). Disabling it.")
            self.__show_window = False

    def destroy(self):
        self.__stop.set()
        if self.__thread is not None:
            self.__thread.join(timeout=5)
            self.__thread = None
