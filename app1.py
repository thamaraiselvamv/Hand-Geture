import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import sqlite3
import hashlib
from datetime import datetime
import time

# [Database setup and authentication functions remain unchanged]
def setup_database():
    conn = sqlite3.connect('user_data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, 
                  created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS activity_log
                 (id INTEGER PRIMARY KEY, user_id INTEGER, action TEXT, 
                  timestamp TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    return conn, c

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(username, password, db_cursor):
    hashed_password = hash_password(password)
    db_cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", 
                     (username, hashed_password))
    result = db_cursor.fetchone()
    return result[0] if result else None

def register_user(username, password, db_conn, db_cursor):
    try:
        hashed_password = hash_password(password)
        db_cursor.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                         (username, hashed_password, datetime.now()))
        db_conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def log_activity(user_id, action, db_conn, db_cursor):
    if user_limits(user_id, db_conn, db_cursor):
        if user_id:
            db_cursor.execute("INSERT INTO activity_log (user_id, action, timestamp) VALUES (?, ?, ?)",
                            (user_id, action, datetime.now()))
            db_conn.commit()

def user_limits(user_id, db_conn, db_cursor):
    db_cursor.execute("SELECT COUNT(*) FROM activity_log WHERE user_id = ? AND timestamp > datetime('now', '-1 hour')", (user_id,))
    count = db_cursor.fetchone()[0]
    return count < 100

# Enhanced Voice Feedback
class VoiceFeedback:
    def __init__(self, speech_rate=150):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', speech_rate)
        self.engine.setProperty('volume', 1.0)  # Max volume for clarity
        self.lock = threading.Lock()
        self.last_spoken = None  # Track last spoken text to avoid repetition
        
    def speak(self, text):
        if text != self.last_spoken:  # Avoid repeating the same announcement
            with self.lock:
                try:
                    self.engine.stop()  # Stop any ongoing speech
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.last_spoken = text
                except RuntimeError:
                    pass  # Handle cases where engine is busy
    
    def set_rate(self, rate):
        self.engine.setProperty('rate', rate)

# Enhanced Gesture Recognition with Realistic Phone
class GestureRecognition:
    def __init__(self, sensitivity=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=sensitivity,
            min_tracking_confidence=sensitivity
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        self.phone_frame = self.create_realistic_phone_frame()
        
        self.brightness_level = 50
        self.volume_level = 50
        self.call_state = "No Call"
        self.screen_state = "On"
        self.emergency_mode = False
        
        self.prev_gesture = None
        self.gesture_cooldown = 0
        self.gesture_history = []

    def create_realistic_phone_frame(self):
        # Create a more detailed phone frame (800x400 with RGBA for transparency)
        frame = np.zeros((800, 400, 4), dtype=np.uint8)
        
        # Phone body (metallic gray with slight gradient)
        cv2.rectangle(frame, (40, 20), (360, 780), (80, 80, 80, 255), -1)  # Main body
        for i in range(40, 360):  # Gradient effect
            alpha = (i - 40) / 320
            color = (int(80 + 20 * alpha), int(80 + 20 * alpha), int(80 + 20 * alpha), 255)
            cv2.line(frame, (i, 20), (i, 780), color)
        
        # Screen (black with subtle border)
        cv2.rectangle(frame, (60, 60), (340, 740), (0, 0, 0, 255), -1)
        cv2.rectangle(frame, (55, 55), (345, 745), (50, 50, 50, 255), 5)  # Bezel
        
        # Home button (circular with shadow)
        cv2.circle(frame, (200, 760), 25, (60, 60, 60, 255), -1)
        cv2.circle(frame, (200, 760), 20, (100, 100, 100, 255), -1)
        
        # Top notch (modern phone design)
        cv2.rectangle(frame, (150, 20), (250, 40), (0, 0, 0, 255), -1)
        cv2.circle(frame, (180, 40), 5, (20, 20, 20, 255), -1)  # Front camera
        
        # Side buttons (volume and power)
        cv2.rectangle(frame, (355, 200), (360, 250), (60, 60, 60, 255), -1)  # Power
        cv2.rectangle(frame, (40, 150), (45, 200), (60, 60, 60, 255), -1)  # Volume up
        cv2.rectangle(frame, (40, 210), (45, 260), (60, 60, 60, 255), -1)  # Volume down
        
        return frame

    def process_gesture(self, gesture, voice_feedback):
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return "None"
        
        if gesture != self.prev_gesture and gesture != "None":
            self.gesture_cooldown = 10
            self.prev_gesture = gesture
            self.gesture_history.append((gesture, datetime.now().strftime("%H:%M:%S")))
            if len(self.gesture_history) > 5:
                self.gesture_history.pop(0)

            percentage_change = 10
            if gesture == "Brightness Up" and self.brightness_level < 100:
                self.brightness_level = min(100, self.brightness_level + percentage_change)
                voice_feedback.speak(f"Brightness up to {self.brightness_level} percent")
            elif gesture == "Brightness Down" and self.brightness_level > 0:
                self.brightness_level = max(0, self.brightness_level - percentage_change)
                voice_feedback.speak(f"Brightness down to {self.brightness_level} percent")
            elif gesture == "Volume Up" and self.volume_level < 100:
                self.volume_level = min(100, self.volume_level + percentage_change)
                voice_feedback.speak(f"Volume up to {self.volume_level} percent")
            elif gesture == "Volume Down" and self.volume_level > 0:
                self.volume_level = max(0, self.volume_level - percentage_change)
                voice_feedback.speak(f"Volume down to {self.volume_level} percent")
            elif gesture == "Answer Call" and self.call_state == "Incoming Call":
                self.call_state = "Call Answered"
                voice_feedback.speak("Attending call")
            elif gesture == "Reject Call" and self.call_state == "Incoming Call":
                self.call_state = "Call Rejected"
                voice_feedback.speak("Rejecting call")
            elif gesture == "Screen On":
                self.screen_state = "On"
                voice_feedback.speak("Screen on")
            elif gesture == "Screen Off":
                self.screen_state = "Off"
                voice_feedback.speak("Screen off")
            elif gesture == "Emergency Call":
                self.emergency_mode = True
                self.call_state = "Emergency Call"
                voice_feedback.speak("Making emergency call")
            
            return gesture
        elif self.call_state == "Incoming Call" and gesture == "None":
            voice_feedback.speak("Call coming")  # Announce incoming call even if no gesture
        return "None"

    def detect_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        gesture = "None"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
                
                for tip_id in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, landmarks[tip_id], 10, (0, 255, 0), -1)
                
                thumb_up = landmarks[4][0] > landmarks[3][0] if landmarks[4][0] > landmarks[0][0] else landmarks[4][1] < landmarks[3][1]
                index_up = landmarks[8][1] < landmarks[6][1]
                middle_up = landmarks[12][1] < landmarks[10][1]
                ring_up = landmarks[16][1] < landmarks[14][1]
                pinky_up = landmarks[20][1] < landmarks[18][1]
                
                finger_states = [
                    f"Thumb: {'Up' if thumb_up else 'Down'}",
                    f"Index: {'Up' if index_up else 'Down'}",
                    f"Middle: {'Up' if middle_up else 'Down'}",
                    f"Ring: {'Up' if ring_up else 'Down'}",
                    f"Pinky: {'Up' if pinky_up else 'Down'}"
                ]
                for i, state in enumerate(finger_states):
                    cv2.putText(frame, state, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "Brightness Up"
                elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "Brightness Down"
                elif not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
                    gesture = "Volume Up"
                elif not thumb_up and not index_up and not middle_up and ring_up and pinky_up:
                    gesture = "Volume Down"
                elif thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "Answer Call"
                elif not thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
                    gesture = "Reject Call"
                elif thumb_up and index_up and middle_up and ring_up and pinky_up:
                    gesture = "Screen On"
                elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "Screen Off"
                elif thumb_up and not index_up and middle_up and not ring_up and not pinky_up:
                    gesture = "Emergency Call"
                
                cv2.putText(frame, gesture, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame, gesture

    def render_phone_screen(self, screen_w, screen_h):
        screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        if self.screen_state == "On":
            cv2.rectangle(screen, (0, 0), (screen_w, 30), (40, 40, 40), -1)  # Status bar
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(screen, time.strftime("%H:%M"), (10, 20), font, 0.5, (255, 255, 255), 1)
            
            content_y = 40
            cv2.putText(screen, f"Brightness: {self.brightness_level}%", (10, content_y + 30), font, 0.7, (255, 255, 255), 2)
            cv2.rectangle(screen, (10, content_y + 40), (10 + int(self.brightness_level * 2), content_y + 50), (0, 255, 255), -1)
            
            cv2.putText(screen, f"Volume: {self.volume_level}%", (10, content_y + 100), font, 0.7, (255, 255, 255), 2)
            cv2.rectangle(screen, (10, content_y + 110), (10 + int(self.volume_level * 2), content_y + 120), (0, 255, 0), -1)
            
            cv2.putText(screen, f"Call: {self.call_state}", (10, content_y + 170), font, 0.7, (255, 255, 255), 2)
            
            if self.emergency_mode:
                cv2.rectangle(screen, (0, 0), (screen_w, screen_h), (0, 0, 255), 10)
                cv2.putText(screen, "EMERGENCY MODE", (screen_w//4, screen_h//2), font, 1, (255, 255, 255), 3)
                
            if self.call_state == "Incoming Call":
                cv2.circle(screen, (screen_w//4, screen_h - 100), 40, (0, 255, 0), -1)
                cv2.putText(screen, "Answer", (screen_w//4 - 30, screen_h - 90), font, 0.7, (255, 255, 255), 2)
                cv2.circle(screen, (3*screen_w//4, screen_h - 100), 40, (255, 0, 0), -1)
                cv2.putText(screen, "Reject", (3*screen_w//4 - 30, screen_h - 90), font, 0.7, (255, 255, 255), 2)
        return screen

    def overlay_phone_frame(self, frame):
        h, w = frame.shape[:2]
        phone_h, phone_w = int(h * 0.9), int(w * 0.4)
        phone_frame_resized = cv2.resize(self.phone_frame.copy(), (phone_w, phone_h))
        
        screen_x, screen_y = int(phone_w * 0.1), int(phone_h * 0.1)
        screen_w, screen_h = int(phone_w * 0.8), int(phone_h * 0.7)
        
        screen_content = self.render_phone_screen(screen_w, screen_h)
        
        if self.screen_state == "On":
            y_offset = int(phone_h * 0.1)
            x_offset = int(phone_w * 0.1)
            phone_frame_resized[y_offset:y_offset+screen_h, x_offset:x_offset+screen_w, 0:3] = screen_content
        
        x_offset, y_offset = w - phone_w - 10, (h - phone_h) // 2
        
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
        alpha_phone = phone_frame_resized[:, :, 3] / 255.0
        alpha_phone = np.repeat(alpha_phone[:, :, np.newaxis], 3, axis=2)
        
        for c in range(0, 3):
            frame_rgba[y_offset:y_offset+phone_h, x_offset:x_offset+phone_w, c] = \
                frame_rgba[y_offset:y_offset+phone_h, x_offset:x_offset+phone_w, c] * (1 - alpha_phone[:, :, c]) + \
                phone_frame_resized[:, :, c] * alpha_phone[:, :, c]
        
        return cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

def main():
    st.set_page_config(layout="wide")
    st.title("Accessible Phone Control with Hand Gestures")
    
    conn, cursor = setup_database()
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.session_state.user_id is None:
        st.subheader("Login / Register")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key="login_button"):
                user_id = login_user(login_username, login_password, cursor)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = login_username
                    log_activity(user_id, "User logged in", conn, cursor)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            if st.button("Register", key="register_button"):
                if reg_password != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if register_user(reg_username, reg_password, conn, cursor):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")
    else:
        st.subheader(f"Welcome, {st.session_state.username}!")
        
        with st.expander("How to use hand gestures"):
            st.markdown("""
            ### Hand Gesture Guide:
            - **Thumb only up**: Brightness Up
            - **Index finger only up**: Brightness Down
            - **Index & Middle fingers up**: Volume Up
            - **Ring & Pinky fingers up**: Volume Down
            - **Thumb & Index finger up**: Answer Call
            - **Pinky finger only up**: Reject Call
            - **All fingers up**: Screen On
            - **All fingers down (fist)**: Screen Off
            - **Thumb & Middle fingers up**: Emergency Call
            """)
        
        with st.expander("Settings"):
            sensitivity = st.slider("Gesture Sensitivity", 0.5, 0.9, 0.7, 0.05)
            speech_rate = st.slider("Voice Feedback Speed", 100, 200, 150, 10)
        
        voice_feedback = VoiceFeedback(speech_rate)
        gesture_rec = GestureRecognition(sensitivity)
        
        if st.button("Logout"):
            log_activity(st.session_state.user_id, "User logged out", conn, cursor)
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        
        with st.expander("Activity Log"):
            cursor.execute("SELECT action, timestamp FROM activity_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", 
                          (st.session_state.user_id,))
            activities = cursor.fetchall()
            for action, timestamp in activities:
                st.write(f"{timestamp}: {action}")
        
        if st.button("Simulate Incoming Call"):
            gesture_rec.call_state = "Incoming Call"
            voice_feedback.speak("Call coming")
            log_activity(st.session_state.user_id, "Simulated incoming call", conn, cursor)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Live Camera Feed")
            video_placeholder = st.empty()
            start_stop = st.button("Stop Camera" if st.session_state.running else "Start Camera")
            if start_stop:
                st.session_state.running = not st.session_state.running
                if st.session_state.running:
                    log_activity(st.session_state.user_id, "Started gesture detection", conn, cursor)
                else:
                    log_activity(st.session_state.user_id, "Stopped gesture detection", conn, cursor)
        
        with col2:
            st.subheader("Phone Display")
            phone_placeholder = st.empty()
            st.subheader("Phone Status")
            status_placeholder = st.empty()
            st.subheader("Recent Gestures")
            gesture_history_placeholder = st.empty()
        
        if st.session_state.running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to access webcam. Please check your camera.")
                st.session_state.running = False
                st.rerun()
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame")
                    st.session_state.running = False
                    break
                
                frame = cv2.flip(frame, 1)
                processed_frame, detected_gesture = gesture_rec.detect_gesture(frame)
                active_gesture = gesture_rec.process_gesture(detected_gesture, voice_feedback)
                
                if active_gesture != "None":
                    log_activity(st.session_state.user_id, f"Performed gesture: {active_gesture}", conn, cursor)
                
                video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                
                phone_frame = gesture_rec.overlay_phone_frame(np.zeros_like(frame))
                phone_placeholder.image(phone_frame, channels="BGR", use_container_width=True)
                
                status_info = f"""
                ### Phone Status:
                - **Brightness:** {gesture_rec.brightness_level}%
                - **Volume:** {gesture_rec.volume_level}%
                - **Call Status:** {gesture_rec.call_state}
                - **Screen:** {gesture_rec.screen_state}
                - **Emergency Mode:** {'On' if gesture_rec.emergency_mode else 'Off'}
                """
                status_placeholder.markdown(status_info)
                
                gesture_history_text = "### Recent Gestures:\n" + "\n".join([f"{time}: {gest}" for gest, time in gesture_rec.gesture_history])
                gesture_history_placeholder.markdown(gesture_history_text)
                
                time.sleep(0.03)
            
            cap.release()
    
    conn.close()

if __name__ == "__main__":
    main()