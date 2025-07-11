import cv2
import os

# Parameters
save_dir = 'dataset'  # Folder where images will be saved
img_size = (224, 224)  # Resize images to this size for model training

# ROI dimensions centered in 1280x720 frame
roi_start = (490, 210)  # top-left corner of box
roi_end = (790, 510)    # bottom-right corner of box

# Labels for classification
labels = ['rock', 'paper', 'sczr', 'none']

# Create directories for each class if they don't exist
for label in labels:
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

print("Press 'r' for Rock, 'p' for Paper, 's' for Scissors, 'n' for None, or 'q' to quit.")

# Counters to keep filenames unique
counter = {label: 0 for label in labels}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)

    # Extract ROI and resize
    roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
    resized_roi = cv2.resize(roi, img_size)

    # Display frame
    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # Handle keypresses
    if key == ord('r'):
        label = 'rock'
    elif key == ord('p'):
        label = 'paper'
    elif key == ord('s'):
        label = 'sczr'
    elif key == ord('n'):
        label = 'none'
    elif key == ord('q'):
        breakq
    else:
        label = None

    # Save image if a label key was pressed
    if label:
        filename = os.path.join(save_dir, laqbel, f"{label}_{counter[label]}.png")
        cv2.imwrite(filename, resized_roi)
        counter[label] += 1
        print(f"[Saved] {filename}")

cap.release()
cv2.destroyAllWindows()
