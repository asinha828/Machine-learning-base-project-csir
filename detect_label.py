import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs

from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np

detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    try:
        image = Image.open(filename).convert('RGB')
        pixels = np.asarray(image)

        results = detector.detect_faces(pixels)
        if not results:
            return None

        x1, y1, width, height = results[0]['box']
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face).resize(required_size)
        return np.asarray(image)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
