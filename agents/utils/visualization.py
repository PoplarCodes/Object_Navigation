import cv2
import numpy as np

def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_name, legend):
    #vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
    """Initialize a blank visualization image with three panels."""
    vis_image = np.ones((655, 1660, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    # Observations title
    text = "Observations (Goal: {})".format(goal_name)
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    #textX = (640 - textsize[0]) // 2 + 15
    textX = 15 + (640 - textsize[0]) // 2
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # Predicted semantic map title
    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    #textX = 640 + (480 - textsize[0]) // 2 + 30
    pred_start = 670
    textX = pred_start + (480 - textsize[0]) // 2
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    # Ground-truth semantic map title
    text = "Ground-Truth Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    gt_start = 1165
    textX = gt_start + (480 - textsize[0]) // 2
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)
    # draw outlines for three panels
    color = [100, 100, 100]
    # Observation panel
    vis_image[49, 15:655] = color
    #vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[530, 15:655] = color

    # Predicted map panel
    vis_image[49, 670:1150] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    #vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color

    # draw legend
    # Ground-truth map panel
    vis_image[49, 1165:1645] = color
    vis_image[50:530, 1164] = color
    vis_image[50:530, 1645] = color
    vis_image[530, 1165:1645] = color

    # draw legend centered at the bottom
    lx, ly, _ = legend.shape
    #vis_image[537:537 + lx, 155:155 + ly, :] = legend
    legend_x = (vis_image.shape[1] - ly) // 2
    vis_image[537:537 + lx, legend_x:legend_x + ly, :] = legend

    return vis_image
