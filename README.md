
Based on [CLIP prefix captioning](https://github.com/rmokady/CLIP_prefix_caption)


### Download weights

Please download the weights from [google drive](https://drive.google.com/drive/folders/1t25Rga6vjUec23W1UsfmnBzI5YvP0l2q?usp=sharing) and place them in the `weights` folder.

There are pretrained weights from [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) project as well as weights obtained by training on a private dataset. The dataset consists of frames from mobile video game ads along with their human-generated captions. The training was conducted using the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) code.

### Setup
You can run this project using either Conda or Docker.

#### Conda
Create and activate the Conda environment:
``` bash
conda env create -f environment.yml
conda activate video_caption
```

#### Docker
You can run scripts in a Docker container. The last line below allows you to enter the bash shell in the Docker container:
```bash
docker build -t video_caption .
docker run --rm -it --entrypoint bash video_caption
```
Please note, there will be an issue with displaying frames during video processing when using Docker (`--display` argument will not work).

### Usage

You can download a promo clip of a farming mobile video game by this [link](https://drive.google.com/file/d/1Y1WA0ga6rfrRTpesQEceAAxKBMfXbVFE/view?usp=sharing) to test the captioning model with different weights.

Run the script with the following command:

``` bash
python main.py [--input_file INPUT_FILE] [--output_file OUTPUT_FILE] [--model_type MODEL_TYPE] [--display]
```

**Arguments**:

- `--input_file`: Path to the input video file. (Default: test_promo_video.mp4)
  
- `--output_file`: Path to save the output video file. If not specified, the output file will be saved as `_output/captioned_{input_file_name}_{model_type}.mp4`.

- `--model_type`: Type of model to use for prediction. Choices are coco, conceptual-captions, trained_mlp, trained_transformer. (Default: trained_mlp)

- `--display`: If this flag is set, the video frames will be displayed during processing. (Default: False)