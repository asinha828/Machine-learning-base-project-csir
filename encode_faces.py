import cv2
import os
import uuid
import time

# Path where images will be stored
IMAGES_PATH = os.path.join('datasets', 'dataset')
os.makedirs(IMAGES_PATH, exist_ok=True)

# Number of images to capture
number_images = 5

# Try using default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam. Try changing the index.")
    exit()

for imgnum in range(number_images):
    print(f'📸 Collecting image {imgnum + 1}/{number_images}')
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ Failed to capture frame.")
        break

    # Save image with unique name
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)

    # Try to display the frame
    try:
        cv2.imshow('Image Collection', frame)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print("⚠️ GUI not supported. Skipping display.")
        time.sleep(0.5)

cap.release()
try:
    cv2.destroyAllWindows()
except:
    pass

print("✅ Image collection complete.")
