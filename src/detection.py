import numpy as np
import cv2
import torch
from argparse import ArgumentParser
import os


def resize(image, size=640):
    height, width, channels = image.shape

    if height > width:
        new_height = size
        new_width = round((width / height) * size)
    else:
        new_width = size
        new_height = round((height / width) * size)
    image = cv2.resize(image, (new_width, new_height))
    return image


def read_image_with_resize(file, size=640):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, size=size)
    return img


def add_rect(image, xmin, ymin, xmax, ymax, color=(255, 0, 0), thickness=2):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    return image


# Model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="/home/thinkin-machine/VSCodeWorkspaces/ANPR_V2/out/detection.pt",
    force_reload=True,
)


def round_all(array):
    return [round(elm) for elm in array]


def detect(image: np.ndarray):
    # Inference
    results = model(image)
    results = results.pandas().xyxy
    res = []
    for pos in results[0].iterrows():
        tmp = round_all(pos[1][:4].tolist())
        res.append(tmp)

    return res


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
        type=str,
        help="path to image on which prediction will be made",
    )

    args = parser.parse_args()

    assert os.path.exists(args.image), f"given path {args.image} does not exists"

    im = read_image_with_resize()(args.image)
    results = detect(im)

    for pos in results:
        im = add_rect(im, *pos)
    cv2.imwrite("result.jpg", im)
