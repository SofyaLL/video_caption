import os
import traceback
from collections import Counter

import clip
import cv2
import numpy as np
import PIL.Image
import skimage.io as io
import torch
import torch.nn.functional as nnf
from PIL import Image
from tqdm import tqdm, trange
from transformers import GPT2Tokenizer

from utils import ClipCaptionModel, ClipCaptionPrefix, get_name

device = "cpu"
CPU = torch.device("cpu")

prefix_length = 10
model_path = "/Users/sonya/PycharmProjects/CLIP_prefix_caption/pretrained_models/coco_weights.pt"

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = ClipCaptionModel(prefix_length)
model.load_state_dict(torch.load(model_path, map_location=CPU))
model = model.eval()
model = model.to(device)


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def predict(image: Image.Image):
    image = preprocess(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix


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
            generated_text_prefix = predict(frame_pil)
            print(generated_text_prefix)
            result_list.append(generated_text_prefix)
            print(generated_text_prefix.capitalize(), file=output_file)

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
