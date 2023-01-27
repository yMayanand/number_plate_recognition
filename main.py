from argparse import ArgumentParser
import os

import cv2
from detection import detect, read_image_with_resize, add_rect
from recognition import recognize, add_text


def extract_number_plate(image, box):
    xmin, ymin, xmax, ymax = box
    return image[ymin:ymax, xmin:xmax, :]


def read_number_plate(image):
    orig_image = image

    boxes = detect(image)

    texts = []
    for box in boxes:
        num_plate = extract_number_plate(orig_image, box)
        text = recognize(num_plate)
        texts.append(text)
    return boxes, texts


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

    im = read_image_with_resize(args.image)

    boxes, texts = read_number_plate(im)
    print(texts)
    for box, text in zip(boxes, texts):
        im = add_rect(im, *box)
        im = add_text(im, text, box)

    cv2.imwrite("result.jpg", im)
