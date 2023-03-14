import cv2
from transformers import GPT2Tokenizer
import clip
from util import ClipCaptionPrefix
import os
from collections import Counter

import skimage.io as io
import PIL.Image
import torch
device = 'cpu'
CPU = torch.device('cpu')
from tqdm import tqdm, trange
import torch.nn.functional as nnf

prefix_length = 10
model_path = './transformer_weights.pt'

clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 40

model = ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=640,
                                  num_layers=8, mapping_type='transformer')
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
        temperature=1.,
        stop_token: str = '.',
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
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
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


def predict(path_img):
    image = io.imread(path_img)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix


if __name__ == '__main__':
    for n in range(1, 5):
    # n = 1
        path_video = f'./ozon_videos/video-{n}-of-4.mp4'
        path_txt = f'./transformer_txt_ozon/video-{n}-of-4.txt'
        path_video_new = f'./new_videos_ozon/cap-video-{n}-of-4.mp4'
        cap = cv2.VideoCapture(path_video)
        fps = cap.get(5)
        width = cap.get(3)
        height = cap.get(4)

        count_frame = 0
        result_list = []
        output_file = open(path_txt, 'w')

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frameSize = (int(width), int(height))
        # x, y, w, h = 185, 160, 346, 180
        # bb = ((x, y), (x + w, y + h))

        new_video = cv2.VideoWriter(path_video_new, fourcc=fourcc, fps=fps, apiPreference=0,
                                    frameSize=frameSize)

        COLOR = (255, 255, 255)

        while True:
            ret, frame = cap.read()
            path_img = f'./frames/{count_frame}.jpg'

            if not ret:
                break

            try:
                # frame_cut = frame[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]
                cv2.imwrite(path_img, frame)
                generated_text_prefix = predict(path_img)
                print(generated_text_prefix)
                result_list.append(generated_text_prefix)
                print(generated_text_prefix.capitalize(), file=output_file)

                cv2.putText(frame, generated_text_prefix, (50, int(height) - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            2, COLOR, 2)
                # cv2.rectangle(frame, (int(bb[0][0]), int(bb[0][1])), (int(bb[1][0]), int(bb[1][1])),
                #               COLOR, 1)
            except:
                pass

            new_video.write(frame)
            count_frame += 1

        print(Counter(result_list), file=output_file)
        output_file.close()

