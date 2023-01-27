# Automatic Number Plate Recognition

## How to use
* Using python api 
  
  `git clone https://github.com/yMayanand/number_plate_recognition.git`

  `cd number_plate_recognition`
  ```python
  from main import read_number_plate
  import cv2

  image = cv2.imread('test.jpg')
  # predicts bounding boxes and number plate of images
  boxes, texts = read_number_plate() 
  ```

* Using Command line
  
  `git clone https://github.com/yMayanand/number_plate_recognition.git`

  `cd number_plate_recognition`

  `python main.py --image add_here_path_to_your_image`

  result will be saved as `result.jpg`

  