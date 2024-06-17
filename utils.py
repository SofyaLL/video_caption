import os
import textwrap

import cv2
import numpy as np

fontsize = 1
font_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX

white_color = (255, 255, 255)
black_color = (0, 0, 0)


def add_banner_text_to_frame(frame, text, banner_height):
    wrapped_text = textwrap.wrap(text, width=30)

    frame_width = frame.shape[1]
    banner = np.ones([banner_height, frame_width, 3], dtype=np.uint8)
    banner.fill(white_color[0])

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, fontsize, font_thickness)[0]

        gap = textsize[1] + 10

        y = int((banner.shape[0] + textsize[1]) / 4) + i * gap
        x = int((banner.shape[1] - textsize[0]) / 2)

        cv2.putText(banner, line, (x, y), font, fontsize, black_color, font_thickness, lineType=cv2.LINE_AA)

    new_frame = np.vstack((frame, banner))

    return new_frame


def get_name(path):
    return os.path.splitext(os.path.basename(path))[0]
