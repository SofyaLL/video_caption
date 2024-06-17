
Based on [CLIP prefix captioning](https://github.com/rmokady/CLIP_prefix_caption)

```
conda env create -f environment.yml
conda activate clip_prefix_caption
```


### Usage

Run the script with the following command:

``` bash
python main.py [--input_file INPUT_FILE] [--output_file OUTPUT_FILE] [--model_type MODEL_TYPE] [--display]
```

**Arguments**:

- `--input_file`: Path to the input video file. (Default: _input/township-10477v1_part.mp4)
  
- `--output_file`: Path to save the output video file. If not specified, the output file will be saved as `_output/captioned_{input_file_name}_{model_type}.mp4`.
- `--model_type`: Type of model to use for prediction. Choices are coco, conceptual-captions, trained_mlp, trained_transformer. (Default: trained_mlp)

- `--display`: If this flag is set, the video frames will be displayed during processing. (Default: False)