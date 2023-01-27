from argparse import ArgumentParser
import pafy
import cv2
from main import read_number_plate
from recognition import add_text
from detection import add_rect


def read_from_youtube(URL):

    # youtube url
    url = URL
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    cap = cv2.VideoCapture(best.url)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            continue

        preds = read_number_plate(image)

        if preds:
            for box, text in zip(*preds):
                image = add_rect(image, *box)
                image = add_text(image, text, box)

        cv2.imshow("ANPR", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    # result.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--link",
        default=None,
        type=str,
        help="link of youtube video to read",
    )

    args = parser.parse_args()

    read_from_youtube(args.link)
