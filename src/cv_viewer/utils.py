import cv2
import numpy as np
import pyzed.sl as sl

id_colors = [(232, 176, 59),
             (175, 208, 25),
             (102, 205, 105),
             (185, 0, 255),
             (99, 107, 252)]

def render_object(object_data, is_tracking_on):
    """Finds if the tracking of object_data is in a valid state.
    Parameters:
        object_data (sl.ObjectData): The object containing the data of a detected object such as its bounding_box, label, id and its 3D position.
        is_tracking_on (bool): Represents if the tracking is activated or not
    Returns:
        bool: The validity of the tracking of object_data
    """
    if is_tracking_on:
        return object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK
    else:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (
                    object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF)


def generate_color_id_u(idx):
    """Generates a color in BGRA format.

    It finds the coresponding color in the id_colors using idx. If idx is < 0 a default color is returned.
    Parameters:
        idx (int): The index for the color
    Returns:
        list[int]: BGRA format of the color
    """
    arr = []
    if idx < 0:
        arr = [236, 184, 36, 255]
    else:
        color_idx = idx % 5
        arr = [id_colors[color_idx][0], id_colors[color_idx][1], id_colors[color_idx][2], 255]
    return arr


def draw_vertical_line(left_display, start_pt, end_pt, clr, thickness):
    """Draws short line segments near the start and end of a line between two points.

    The method divides the line between `start_pt` and `end_pt` into `n_steps` equal parts
    and draws only the first and last segments, leaving the central portion blank.
    Parameters:
        left_display (np.array): The image to draw in
        start_pt (list[int] or tupple): The starting coordinates of the line
        end_pt (list[int] or tupple): The ending coordinates of the line
        clr (list[int] or tupple): The color of the line
        thickness (int): The thickness of the line in pixels
    Returns:
        None
    """
    n_steps = 7
    pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps
        , ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
    pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps
        , (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]

    cv2.line(left_display, (int(start_pt[0]), int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
    cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]), int(end_pt[1])), clr, thickness)
