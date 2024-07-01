import argparse
import os

import cv2
from loguru import logger
from PIL import Image

from predict import Predictor
from utils import add_banner_text_to_frame, get_name

default_video = "test_promo_video.mp4"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Caption a video")
    parser.add_argument("--input_file", type=str, default=default_video, help="Path to input video")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save tqhe output video")
    parser.add_argument(
        "--model_type",
        type=str,
        default="trained_mlp",
        choices=["coco", "conceptual-captions", "trained_mlp", "trained_transformer"],
        help="Type of model to use for prediction",
    )
    parser.add_argument("--display", action="store_true", help="Display video frames during processing")
    return parser.parse_args()


def main():
    args = parse_arguments()
    video_name = get_name(args.input_file)
    if args.output_file:
        new_video_path = args.output_file
    else:
        os.makedirs("_output", exist_ok=True)
        new_video_path = f"_output/captioned_{video_name}_{args.model_type}.mp4"

    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {new_video_path}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Display: {args.display}")

    predictor = Predictor(model_type=args.model_type)

    cap = cv2.VideoCapture(args.input_file)
    fps = cap.get(5)
    width = int(cap.get(3))
    height = int(cap.get(4))

    count_frame = 0

    banner_height = 200
    new_frame_size = (width, height + banner_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    new_video = cv2.VideoWriter(
        new_video_path, fourcc=fourcc, fps=fps, apiPreference=0, frameSize=new_frame_size
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            generated_text = predictor.predict(frame_pil, use_beam_search=False)
            logger.info(f"Frame {count_frame}: {generated_text}")
        except Exception as e:
            logger.error(e)
            # print(traceback.format_exc())

        new_frame = add_banner_text_to_frame(frame, generated_text, banner_height)

        if args.display:
            cv2.imshow("Frame", new_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        new_video.write(new_frame)
        count_frame += 1

    cap.release()
    new_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
