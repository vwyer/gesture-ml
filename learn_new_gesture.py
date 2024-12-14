import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time


# Global variables
GESTURE_TYPE ="nothing" # Change this to the desired gesture type
FRAME_COUNT = 0 # Start capturing frames from 0
MAX_FRAMES = 1000 # Maximum number of frames to capture
SPEED = 0.5 # Time interval between each frame capture (in seconds)


# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Database setup with separate columns for each landmark
conn = sqlite3.connect('hand_gestures.db')
cursor = conn.cursor()

# Create a table with separate columns for each landmark coordinate
columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "gesture_type TEXT", "frame_id INTEGER"]
for hand in ["left", "right"]:
    for point in ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                  "index_mcp", "index_pip", "index_dip", "index_tip",
                  "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                  "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                  "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]:
        columns.append(f"{hand}_{point}_x REAL")
        columns.append(f"{hand}_{point}_y REAL")
columns.append("timestamp TEXT")

# Create table with the defined schema
cursor.execute(f'''CREATE TABLE IF NOT EXISTS gesture_data ({", ".join(columns)})''')
conn.commit()


def save_to_database(gesture_type, frame_id, left_hand_data, right_hand_data):
    """Save each hand's landmarks into separate columns."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    values = [gesture_type, frame_id]
    
    # Add left hand landmarks
    values.extend(left_hand_data if left_hand_data else [None] * 42)
    # Add right hand landmarks
    values.extend(right_hand_data if right_hand_data else [None] * 42)
    # Add timestamp
    values.append(timestamp)

    cursor.execute(f'''INSERT INTO gesture_data 
                       ({", ".join([col.split(" ")[0] for col in columns[1:]])})
                       VALUES ({", ".join(["?" for _ in values])})''',
                   values)
    conn.commit()

def main():
    cap = cv2.VideoCapture(0)
    frame_count = FRAME_COUNT
    max_frames = MAX_FRAMES
    speed = SPEED

    start_time = time.time() 

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Konnte kein Bild erfassen.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        overlay = np.zeros_like(frame) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        # Draw landmarks and connections on the overlay if hands are detected
        left_hand_data = None
        right_hand_data = None

        if hand_results.multi_hand_landmarks:
            for hand_landmarks, hand_label in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_type = hand_label.classification[0].label

                # Collect all landmark coordinates into a flat list of 42 values (x, y pairs)
                hand_data = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

                if hand_type == "Left":
                    left_hand_data = hand_data
                elif hand_type == "Right":
                    right_hand_data = hand_data

                # Draw landmarks and connections without individual point coordinates
                mp_draw.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        
        elapsed_time = time.time() - start_time

        
        if elapsed_time >= speed:
            save_to_database(GESTURE_TYPE, frame_count, left_hand_data, right_hand_data)
            frame_count += 1
            start_time = time.time() 

        # Show frame capture progress at the top left
        cv2.putText(overlay, f"Frames Captured: {frame_count}/{max_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the overlay with hand landmarks and connections only
        cv2.imshow('Gesture Data Capture - Overlay Only', overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
