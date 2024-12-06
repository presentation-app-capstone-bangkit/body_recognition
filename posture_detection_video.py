import cv2
import mediapipe as mp
import numpy as np
import argparse

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate the angle between three points (hip, shoulder, horizontal)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def main(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        total_frames = 0
        good_posture_frames = 0
        bad_posture_frames = 0
        moving_hand_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get frame dimensions
            height, width, _ = frame.shape

            # Dynamically set font scale and thickness based on frame dimensions
            font_scale = width / 1400
            thickness = max(1, int(font_scale * 2))

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process the frame
            results = pose.process(rgb_frame)

            # Convert frame back to BGR for rendering
            rgb_frame.flags.writeable = True
            output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                coords = {
                    name: (int(landmarks[name.value].x * width), int(landmarks[name.value].y * height))
                    for name in [mp_pose.PoseLandmark.LEFT_SHOULDER,
                                 mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                 mp_pose.PoseLandmark.LEFT_HIP,
                                 mp_pose.PoseLandmark.RIGHT_HIP,
                                 mp_pose.PoseLandmark.LEFT_WRIST,
                                 mp_pose.PoseLandmark.RIGHT_WRIST,
                                 mp_pose.PoseLandmark.LEFT_ELBOW,
                                 mp_pose.PoseLandmark.RIGHT_ELBOW]
                }

                # Draw lines between key points
                connections = [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
                ]

                for connection in connections:
                    pt1 = coords.get(connection[0])
                    pt2 = coords.get(connection[1])
                    if pt1 and pt2:
                        cv2.line(output_frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

                # Calculate Shoulder Distance
                shoulder_distance = calculate_distance(coords[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                       coords[mp_pose.PoseLandmark.RIGHT_SHOULDER])

                # Calculate Hip-to-Shoulder Angle
                hip_coords = coords[mp_pose.PoseLandmark.LEFT_HIP]
                shoulder_coords = coords[mp_pose.PoseLandmark.LEFT_SHOULDER]
                # Horizontal reference
                horizontal_coords = (shoulder_coords[0], hip_coords[1])
                hip_to_shoulder_angle = calculate_angle(hip_coords, shoulder_coords, horizontal_coords)
                
                # Detect Hand Movement
                left_wrist_movement = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z
                right_wrist_movement = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
                hands_playing = abs(left_wrist_movement) > 0.09 or abs(right_wrist_movement) > 0.09
                
                if hands_playing:
                    moving_hand_frames += 1
                    # Determine Posture
                    if abs(hip_to_shoulder_angle - 90) >= 84 and abs(hip_to_shoulder_angle - 90) < 94:
                        good_posture_frames += 1
                    else:
                        bad_posture_frames += 1

                total_frames += 1

                # Annotate on the video
                for key, (x, y) in coords.items():
                    cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1, cv2.LINE_AA)

                # Texts to display
                text_lines = [
                    f"Shoulder Distance: {int(shoulder_distance)}",
                    f"Hip-to-Shoulder Angle: {int(abs(hip_to_shoulder_angle - 90))}",
                    f"Posture: {'Good' if abs(hip_to_shoulder_angle - 90) >= 84 and abs(hip_to_shoulder_angle - 90) < 94 else 'Bad'}",
                    f"Hands Playing: {'Yes' if hands_playing else 'No'}",
                ]

                # Calculate the size of the text and background
                y_offset = 20
                for text in text_lines:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_width, text_height = text_size[0]
                    bg_top_left = (20, y_offset - 10)
                    bg_bottom_right = (20 + text_width + 20, y_offset + text_height + 20)
                    cv2.rectangle(output_frame, bg_top_left, bg_bottom_right, (25, 25, 25), -1)
                    cv2.putText(output_frame, text, (30, y_offset + text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (255, 255, 255), thickness, cv2.LINE_AA)
                    y_offset += text_height + 20

            # Display the frame
            cv2.imshow("Pose Detection with Posture Analysis", output_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Final Result Analysis
        if good_posture_frames > bad_posture_frames:
            result = "Your body in the presentation is good."
        else:
            result = "Your body in the presentation is bad."

        print(f"Final Analysis:")
        print(f" - Good Posture Frames: {good_posture_frames}")
        print(f" - Bad Posture Frames: {bad_posture_frames}")
        print(f" - {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Detection and Posture Analysis")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    args = parser.parse_args()
    main(args.video_path)