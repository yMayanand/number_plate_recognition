from argparse import ArgumentParser
from itertools import groupby
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH has to be a multiple of 16"

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(
                "conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
            )
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module("relu{0}".format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module(
            "pooling{0}".format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module(
            "pooling{0}".format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


VOCAB = [
    "BLANK",
    "Z",
    "B",
    "4",
    "X",
    "R",
    "2",
    "U",
    "D",
    "G",
    "Q",
    "S",
    "A",
    "N",
    "K",
    "0",
    "C",
    "J",
    "P",
    "Y",
    "H",
    "7",
    "W",
    "V",
    "5",
    "F",
    "L",
    "8",
    "1",
    "I",
    "T",
    "M",
    "3",
    "O",
    "9",
    "E",
    "6",
]


def add_text(image, text, pos):
    xmin, ymin, xmax, ymax = pos
    image = cv2.putText(
        image,
        text,
        (xmin, ymin - 15),
        cv2.FONT_HERSHEY_COMPLEX,
        0.85,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def greedy_decode(preds):
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    best_chars_collapsed = [k for k, _ in groupby(preds) if k != "BLANK"]
    res = "".join(best_chars_collapsed)
    return res


def read_image(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def idx2char(preds):
    return [VOCAB[idx] for idx in preds]


def post_process(preds):
    # preds shape (seq_len, num_class)
    _, preds = torch.max(preds, dim=1)
    return idx2char(preds.tolist())


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((32, 128)),
        transforms.Normalize(0.5, 0.5),
    ]
)

model = CRNN(32, 1, 37, 512)

state = torch.load("/home/thinkin-machine/VSCodeWorkspaces/ANPR_V2/out/ocr_point08.pt")
model.load_state_dict(state["model"])


def recognize(image):
    model.eval()
    preds = model(transform(image).unsqueeze(0))
    text = post_process(preds[:, 0, :])
    text = greedy_decode(text)
    return text


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

    im = read_image(args.image)

    text = recognize(im)
    print(text)
