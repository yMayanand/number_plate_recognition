from urllib.request import urlretrieve
import os

RECOGNITION_MODEL_PATH = 'https://github.com/yMayanand/number_plate_recognition/releases/download/v1.0/ocr_point08.pt'
DETECTION_MODEL_PATH = 'https://github.com/yMayanand/number_plate_recognition/releases/download/v1.0/detection.pt'

os.makedirs('./out', exist_ok=True)

reco_model = './out/ocr_point08.pt'
det_model = './out/detection.pt'

if not os.path.exists(reco_model) or not os.path.exists(det_model):
    print('🚀 Downloading Model Weights...')
    urlretrieve(RECOGNITION_MODEL_PATH, reco_model)
    urlretrieve(DETECTION_MODEL_PATH, det_model)
else:
    print('Model Weights already there')