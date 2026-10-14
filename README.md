# Custom Object Detection

A workspace for training and running a custom [YOLOv5](https://github.com/ultralytics/yolov5) object detector. It holds the labelled datasets, the class list, and a script that runs detection on a live webcam feed.

## Classes

The custom model is trained on 16 classes (listed in `classes.txt`):

`dog`, `person`, `cat`, `tv`, `car`, `meatballs`, `marinara sauce`, `tomato soup`, `chicken noodle soup`, `french onion soup`, `chicken breast`, `ribs`, `pulled pork`, `hamburger`, `cavity`, `fire`

## What's in the repo

| Path | Contents |
| --- | --- |
| `data/` | Main dataset in YOLO format: `train/`, `valid/` and `test/` splits (each with `images/` and `labels/`), plus top-level `images/` and `labels/` folders with the full pool of annotated images |
| `data2/`, `data3/` | Smaller extra datasets (images and labels) |
| `classes.txt` | Class names, one per line |
| `env/app.py` | Webcam detection script (see below) |
| `yolov5/`, `labelImg/` | Links to the YOLOv5 and labelImg repositories (see the note below) |
| `env/` | A committed Python virtual environment |

Image filenames carry Roboflow's `.rf.` hash suffix, so the datasets appear to be Roboflow exports. Annotations were made or adjusted with [labelImg](https://github.com/HumanSignal/labelImg).

## Running live detection

`env/app.py` opens the default webcam (device 0) and runs two models on each frame:

1. the custom model, loaded from `runs/train/exp3/weights/last.pt`
2. the stock `yolov5s.pt` COCO model

Each model's annotated frame is shown in an OpenCV window. Press **q** to quit.

```bash
pip install torch opencv-python matplotlib numpy
python env/app.py
```

`torch.hub` downloads the YOLOv5 code from GitHub on first run, so you need an internet connection. The trained weights (`runs/` and `yolov5s.pt`) aren't committed, so train your own custom model (or edit the paths at the top of `app.py`) and place `yolov5s.pt` in the folder you run from.

## Training your own

Training follows the standard YOLOv5 workflow: clone YOLOv5, point a dataset YAML at `data/` and the class names in `classes.txt`, and run its `train.py`. The best and last weights land in `runs/train/exp*/weights/`, which is where `app.py` expects them.
