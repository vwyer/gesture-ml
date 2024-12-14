import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import pyautogui 

# Load the trained model
MODEL_PATH = 'gesture_recognition_model.pkl'

# Mapping for gestures (this should match the labels used in the training script)
GESTURE_LABELS = {
    "next": "Next slide", # Gesture label: "next" -> Display text: "Next slide"
    "back": "Previous slide", # Gesture label: "back" -> Display text: "Previous slide"
    "nothing": "Nothing detected", # Gesture label: "nothing" -> Display text: "Nothing detected"
}

# Map gestures to keyboard keys
GESTURE_KEY_MAPPING = {
    "next": "right",       # Simulates pressing the right arrow key
    "back": "left",      # Simulates pressing the left arrow key
}



model = joblib.load(MODEL_PATH)

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Feature columns used for training
FEATURE_COLUMNS = [f"{hand}_{point}_{axis}" for hand in ["left", "right"]
                   for point in ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                                 "index_mcp", "index_pip", "index_dip", "index_tip",
                                 "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                                 "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                                 "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
                   for axis in ["x", "y"]]

def extract_landmarks(hand_landmarks):
    """Extracts the x and y coordinates from Mediapipe landmarks and returns as a flat list."""
    return [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

def main():
    # Customize the text size for the gesture display
    text_size = 2  # Change this value to adjust text size

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not capture image.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        overlay = frame.copy()  # Create an overlay to draw on

        gesture_text = "No Gesture"

        # Initialize empty lists for both hands
        left_hand_landmarks = [0] * 42  # Fill 42 zeros for missing left hand
        right_hand_landmarks = [0] * 42  # Fill 42 zeros for missing right hand

        # Check if any hands are detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, hand_label in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_type = hand_label.classification[0].label

                # Draw the hand landmarks
                mp_draw.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                # Extract landmarks for model prediction
                landmarks = extract_landmarks(hand_landmarks)

                # Assign landmarks to the correct hand
                if hand_type == "Left":
                    left_hand_landmarks = landmarks
                elif hand_type == "Right":
                    right_hand_landmarks = landmarks

        # Create a full feature vector with 84 features (42 for each hand)
        full_feature_vector = left_hand_landmarks + right_hand_landmarks

        # Convert to a DataFrame with the same feature names as during training
        input_df = pd.DataFrame([full_feature_vector], columns=FEATURE_COLUMNS)

        # Predict the gesture using the model
        gesture_type = model.predict(input_df)[0]

        # Map gesture type to label
        gesture_text = GESTURE_LABELS.get(gesture_type, "Unknown Gesture")

        # Check if the gesture has a corresponding key mapping
        if gesture_type in GESTURE_KEY_MAPPING:
            key_to_press = GESTURE_KEY_MAPPING[gesture_type]
            # Simulate key press
            pyautogui.press(key_to_press)

            #insert a delay to prevent multiple key presses
            cv2.waitKey(1000)

        # Display the predicted gesture on the live feed
        cv2.putText(overlay, f"{gesture_text}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the overlay with gesture text
        cv2.imshow('Live Gesture Recognition', overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
