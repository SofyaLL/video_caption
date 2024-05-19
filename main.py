import os
import traceback
from collections import Counter

import cv2
from PIL import Image

from predict import Predictor
from utils import get_name

# 'coco', 'conceptual-captions', 'trained_mlp', 'trained_transformer'
predictor = Predictor(model_type="trained_transformer")


if __name__ == "__main__":
    video_path = "_input/township-10477v1_part.mp4"
    video_name = get_name(video_path)
    txt_path = f"_output/{video_name}.txt"
    frame_paths = f"_output/{video_name}_frames"
    os.makedirs(frame_paths, exist_ok=True)
    new_video_path = f"_output/captioned_{video_name}.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(5)
    width = cap.get(3)
    height = cap.get(4)

    count_frame = 0
    result_list = []
    output_file = open(txt_path, "w")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frameSize = (int(width), int(height))

    new_video = cv2.VideoWriter(new_video_path, fourcc=fourcc, fps=fps, apiPreference=0, frameSize=frameSize)

    COLOR = (255, 255, 255)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            generated_text_prefix = predictor.predict(frame_pil, use_beam_search=False)
            print(generated_text_prefix)

            cv2.putText(
                frame,
                generated_text_prefix,
                (50, int(height) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                COLOR,
                2,
            )
        except:
            print(traceback.format_exc())

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        new_video.write(frame)
        count_frame += 1
        print(f"Processed {count_frame} frames")
