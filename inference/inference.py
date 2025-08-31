import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from try_on.try_on import generate_tryon

IMAGE_FILE = 'example1.jpg'
CLOTH_IMAGE_FILE = 'cloth_example1.jpg'

image = cv2.cvtColor(cv2.imread(os.path.join('..', IMAGE_FILE)), cv2.COLOR_BGR2RGB)
cloth_image = cv2.cvtColor(cv2.imread(os.path.join('..', CLOTH_IMAGE_FILE)), cv2.COLOR_BGR2RGB)

tryon_small = generate_tryon(image, cloth_image).numpy()
tryon_large = generate_tryon(image, cloth_image,  size='large').numpy()

tryon_small = np.clip((tryon_small * 255).astype(np.uint8), 0, 255)
tryon_large = np.clip((tryon_large * 255).astype(np.uint8), 0, 255)

cv2.imwrite('tryon_small.jpg', cv2.cvtColor(tryon_small, cv2.COLOR_RGB2BGR))
cv2.imwrite('tryon_large.jpg', cv2.cvtColor(tryon_large, cv2.COLOR_RGB2BGR))