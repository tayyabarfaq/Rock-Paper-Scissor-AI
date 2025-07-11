# Enhanced Rock Paper Scissors UI
import cv2
import numpy as np
import random
import time
import mediapipe as mp
from keras.models import load_model

model = load_model("rps_my_data_model.h5")
class_names = ['paper', 'rock', 'scissors']

image_paths = {
    'rock': 'Resources/rock.png',
    'paper': 'Resources/paper.png',
    'scissors': 'Resources/scissors.png'
}
move_images = {name: cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), (224, 224)) for name, path in image_paths.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "rock" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "rock"):
        return "You Win!"
    else:
        return "Computer Wins!"

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (1. - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    else:
        background[y:y+h, x:x+w] = overlay
    return background

def draw_text_with_shadow(img, text, position, font, scale, color, thickness):
    x, y = position
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

cap = cv2.VideoCapture(0)
x1, y1, x2, y2 = 100, 100, 324, 324

print("Press 'space' to play or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = hands.process(roi_rgb)

    hand_detected = False
    if result.multi_hand_landmarks:
        hand_detected = True
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)

    if hand_detected:
        img = cv2.resize(roi, (224, 224)).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        label = f"{class_names[class_idx]} ({confidence * 100:.1f}%)" if confidence > 0.75 else "Uncertain"
        draw_text_with_shadow(frame, f"Live: {label}", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    else:
        draw_text_with_shadow(frame, "Show hand inside the box", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    draw_text_with_shadow(frame, "Press SPACE to play", (70, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Rock Paper Scissors - VS Computer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        start_time = time.time()
        countdown_time = 3
        final_prediction = None
        final_confidence = 0

        animation_interval = 0.2
        last_anim_time = time.time()
        cpu_index = 0
        cpu_sequence = ['rock', 'paper', 'scissors']

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            roi = frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            current_time = time.time()
            elapsed = int(current_time - start_time)
            remaining = countdown_time - elapsed

            if remaining > 0:
                result = hands.process(roi_rgb)
                if result.multi_hand_landmarks:
                    for handLms in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)
                    img = cv2.resize(roi, (224, 224)).astype('float32') / 255.0
                    img = np.expand_dims(img, axis=0)
                    prediction = model.predict(img, verbose=0)[0]
                    class_idx = np.argmax(prediction)
                    confidence = prediction[class_idx]
                    if confidence > final_confidence and confidence > 0.75:
                        final_prediction = class_names[class_idx]
                        final_confidence = confidence
                    label = f"{class_names[class_idx]} ({confidence * 100:.1f}%)" if confidence > 0.75 else "Uncertain"
                    draw_text_with_shadow(frame, f"Live: {label}", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if current_time - last_anim_time > animation_interval:
                    cpu_index = (cpu_index + 1) % 3
                    last_anim_time = current_time

                anim_img = move_images[cpu_sequence[cpu_index]]
                frame = overlay_transparent(frame, anim_img, 350, 100)
                draw_text_with_shadow(frame, str(remaining), (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)

            else:
                # Final capture after countdown
                result = hands.process(roi_rgb)
                if result.multi_hand_landmarks:
                    for handLms in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)
                    img = cv2.resize(roi, (224, 224)).astype('float32') / 255.0
                    img = np.expand_dims(img, axis=0)
                    prediction = model.predict(img, verbose=0)[0]
                    class_idx = np.argmax(prediction)
                    confidence = prediction[class_idx]
                    if confidence > final_confidence and confidence > 0.75:
                        final_prediction = class_names[class_idx]
                        final_confidence = confidence

                computer_move = random.choice(class_names)
                frame = overlay_transparent(frame, move_images[computer_move], 350, 100)
                draw_text_with_shadow(frame, "Shoot!", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow("Rock Paper Scissors - VS Computer", frame)
                cv2.waitKey(1000)
                break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Rock Paper Scissors - VS Computer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if final_prediction is None:
            result_frame = np.zeros_like(frame)
            draw_text_with_shadow(result_frame, "Unclear Move!", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Rock Paper Scissors - VS Computer", result_frame)
            cv2.waitKey(2000)
            continue

        player_move = final_prediction
        result_text = get_winner(player_move, computer_move)

        frame = overlay_transparent(frame, move_images[computer_move], 350, 100)
        draw_text_with_shadow(frame, f"You: {player_move}", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        draw_text_with_shadow(frame, f"CPU: {computer_move}", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.rectangle(frame, (130, 430), (500, 480), (0, 0, 0), -1)
        draw_text_with_shadow(frame, result_text, (150, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.imshow("Rock Paper Scissors - VS Computer", frame)
        cv2.waitKey(3000)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
