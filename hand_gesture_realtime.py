import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import flask
from flask import Flask, Response, render_template, request, jsonify
import threading
import base64
import json
import os
import time
import mediapipe as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

# Constants
IMG_SIZE = 224
CLASSES = ['Hi', 'No', 'Ok', 'Talk', 'You']
CONFIDENCE_THRESHOLD = 0.50  # Lowered threshold to 50% for better detection
ROI = {"top": 100, "right": 500, "bottom": 350, "left": 100}  # Adjusted ROI dimensions

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create Flask application
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Global variables
model = None
camera = None
output_frame = None
lock = threading.Lock()
last_prediction = {"class": "", "confidence": 0}
processing_thread = None
camera_running = False

# Load the model
def load_gesture_model():
    global model
    # Check for multiple possible paths
    model_paths = [
    os.path.join('saradhe', 'hand_gesture_resnet50v2_model.h5'),
    'hand_gesture_best_model.h5',
    'hand_gesture_resnet50v2_model.h5',
    os.path.join('models', 'hand_gesture_resnet50v2_model.h5'),
    # Add current directory path
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_gesture_resnet50v2_model.h5'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'hand_gesture_resnet50v2_model.h5')
]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}")
                # Try to avoid memory issues
                tf.keras.backend.clear_session()
                model = load_model(model_path)
                print("Model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    print("ERROR: Could not find model file in any of the expected locations")
    return False

# Initialize webcam
def initialize_camera():
    global camera, camera_running
    
    # If camera is already running, return success
    if camera is not None and camera.isOpened():
        camera_running = True
        return True
    
    # Try multiple camera indices
    for camera_index in range(3):  # Try camera indices 0, 1, 2
        try:
            print(f"Attempting to open camera with index {camera_index}")
            camera = cv2.VideoCapture(camera_index)
            # Set camera properties for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if camera.isOpened():
                # Check if we can read a frame
                ret, frame = camera.read()
                if ret:
                    print(f"Successfully opened camera with index {camera_index}")
                    camera_running = True
                    return True
                else:
                    print(f"Could open camera {camera_index} but couldn't read frame")
                    camera.release()
            else:
                print(f"Failed to open camera with index {camera_index}")
        except Exception as e:
            print(f"Error accessing camera {camera_index}: {e}")
    
    # If we get here, all camera indices failed
    camera = None
    camera_running = False
    print("ERROR: Could not open any webcam")
    return False

# Release camera resources
def release_camera():
    global camera, processing_thread, camera_running
    
    camera_running = False
    time.sleep(0.5)  # Give time for threads to recognize camera_running is False
    
    if camera is not None:
        try:
            camera.release()
            print("Camera released")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        finally:
            camera = None
    
    # Don't attempt to stop the thread - just let it exit naturally
    # when camera_running becomes False

# Process frames and make predictions
def process_frames():
    global camera, output_frame, lock, last_prediction, camera_running
    
    print("Frame processing thread started")
    frame_count = 0
    last_valid_prediction = None
    prediction_history = []
    
    while camera_running:
        if camera is None or not camera.isOpened():
            print("Camera not available in processing thread")
            time.sleep(0.5)
            continue
        
        try:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture image")
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            roi_top, roi_right, roi_bottom, roi_left = ROI["top"], ROI["right"], ROI["bottom"], ROI["left"]
            cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                
                h, w, _ = frame.shape
                x_min, x_max, y_min, y_max = w, 0, h, 0
                
                for landmark in hand.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                
                padding = 30
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                if (x_min >= roi_left and x_max <= roi_right and 
                    y_min >= roi_top and y_max <= roi_bottom):
                    
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    
                    mp_drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    frame_count += 1
                    if frame_count % 2 == 0:
                        # Pass hand landmarks to predict_gesture
                        predicted_class, confidence = predict_gesture(hand_roi, hand)
                        
                        if predicted_class != "Uncertain" and predicted_class != "Error":
                            prediction_history.append(predicted_class)
                            if len(prediction_history) > 5:
                                prediction_history.pop(0)
                            
                            if len(prediction_history) >= 3:
                                from collections import Counter
                                most_common = Counter(prediction_history).most_common(1)[0]
                                if most_common[1] >= 3:
                                    predicted_class = most_common[0]
                                    last_valid_prediction = predicted_class
                        
                        if confidence > CONFIDENCE_THRESHOLD:
                            with lock:
                                last_prediction = {"class": predicted_class, "confidence": float(confidence)}
                        
                        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
                        text = f"{predicted_class}: {confidence:.2f}%"
                        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        # Draw finger states
                        fingers = get_gesture_features(hand)
                        finger_text = "Fingers: " + " ".join([k for k, v in fingers.items() if v])
                        cv2.putText(frame, finger_text, (30, frame.shape[0] - 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        guide_text = f"Current Gesture: {last_valid_prediction if last_valid_prediction else 'None'}"
                        cv2.putText(frame, guide_text, (30, frame.shape[0] - 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Move hand inside the box", (30, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (30, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            guide_text = "Show your hand clearly within the green box"
            cv2.putText(frame, guide_text, (30, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            with lock:
                output_frame = frame.copy()
                
        except Exception as e:
            print(f"Error in processing thread: {e}")
            time.sleep(0.1)
    
    print("Frame processing thread exiting")

# Generate frames for the video feed
def generate_frames():
    global output_frame, lock, camera_running
    
    while True:
        # If camera is not running, yield a blank frame
        if not camera_running:
            # Create a blank frame with "Camera Off" text
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Camera Off", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
            
            # Encode the blank frame
            (flag, encoded_image) = cv2.imencode(".jpg", blank_frame)
            if flag:
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encoded_image) + b'\r\n')
            
            time.sleep(0.5)
            continue
        
        # Check if we have a frame
        with lock:
            if output_frame is None:
                time.sleep(0.1)
                continue
            
            # Make a copy to avoid issues if frame is updated while encoding
            current_frame = output_frame.copy()
        
        # Encode the frame as JPEG
        try:
            # Add quality parameter for better performance
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            (flag, encoded_image) = cv2.imencode(".jpg", current_frame, encode_params)
            if not flag:
                continue
                
            # Yield the output frame in byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_image) + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            time.sleep(0.1)

# Make prediction on ROI
def get_hand_orientation(landmarks):
    # Get wrist and middle finger MCP (base) points
    wrist = landmarks.landmark[0]  # Wrist
    middle_mcp = landmarks.landmark[9]  # Middle finger MCP
    
    # Calculate angle
    angle = np.degrees(np.arctan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x))
    return angle

def get_gesture_features(hand_landmarks):
    # Get key points for gesture recognition
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Get base points
    thumb_base = hand_landmarks.landmark[2]
    index_base = hand_landmarks.landmark[5]
    middle_base = hand_landmarks.landmark[9]
    ring_base = hand_landmarks.landmark[13]
    pinky_base = hand_landmarks.landmark[17]
    
    # Calculate finger states (extended or not)
    fingers_extended = {
        'thumb': (thumb_tip.y < thumb_base.y),
        'index': (index_tip.y < index_base.y),
        'middle': (middle_tip.y < middle_base.y),
        'ring': (ring_tip.y < ring_base.y),
        'pinky': (pinky_tip.y < pinky_base.y)
    }
    
    return fingers_extended

def predict_gesture(roi, hand_landmarks=None):
    if model is None:
        return "Model not loaded", 0
    
    try:
        # Ensure minimum size and aspect ratio
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return "Hand too small", 0
        
        # Get hand orientation and features if landmarks are provided
        if hand_landmarks:
            orientation = get_hand_orientation(hand_landmarks)
            fingers = get_gesture_features(hand_landmarks)
            
            # Use geometric features to help validate predictions
            # For Talk gesture: thumb and pinky should be extended, others closed
            if fingers['thumb'] and not fingers['index'] and not fingers['middle'] and not fingers['ring'] and fingers['pinky']:
                gesture_hint = "Talk"
            # For Hi gesture: all fingers should be extended
            elif all(fingers.values()):
                gesture_hint = "Hi"
            else:
                gesture_hint = None
        
        # Maintain aspect ratio while resizing
        h, w = roi.shape[:2]
        # Pad to square with black
        size = max(h, w)
        square = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = roi
        
        # Resize maintaining square aspect ratio
        roi_resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
        
        # Enhance contrast
        lab = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert to RGB and normalize
        roi_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        roi_normalized = roi_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        roi_input = np.expand_dims(roi_normalized, axis=0)
        
        # Make prediction
        prediction = model.predict(roi_input, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx] * 100
        
        # Get top 2 predictions for debugging
        top_2_idx = np.argsort(prediction[0])[-2:][::-1]
        print("\nTop 2 Predictions:")
        for idx in top_2_idx:
            print(f"{CLASSES[idx]}: {prediction[0][idx]*100:.2f}%")
        
        predicted_class = CLASSES[predicted_class_idx]
        
        # Use geometric hint to validate prediction
        if hand_landmarks and gesture_hint:
            if predicted_class != gesture_hint and confidence < 90:
                if gesture_hint == "Talk" and predicted_class == "Hi":
                    predicted_class = "Talk"
                    confidence = max(confidence, 75)  # Set minimum confidence for geometric validation
        
        # Add minimum confidence threshold
        if confidence < 30:
            return "Uncertain", 0
            
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0

# Flask routes
@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera, processing_thread, camera_running
    
    if camera_running:
        return jsonify({"success": True, "message": "Camera already running"})
    
    if initialize_camera():
        # Set flag before starting thread
        camera_running = True
        
        # Start the processing thread
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
        
        return jsonify({"success": True, "message": "Camera started successfully"})
    else:
        return jsonify({"success": False, "message": "Failed to start camera"})

@app.route('/stop_camera')
def stop_camera():
    global camera_running
    
    if not camera_running:
        return jsonify({"success": True, "message": "Camera already stopped"})
    
    try:
        release_camera()
        return jsonify({"success": True, "message": "Camera stopped successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error stopping camera: {str(e)}"})

@app.route('/get_prediction')
def get_prediction():
    global last_prediction
    
    with lock:
        prediction = last_prediction.copy()
    
    return jsonify({"success": True, "prediction": prediction})

@app.route('/check_camera_status')
def check_camera_status():
    global camera_running
    return jsonify({"running": camera_running})

# New endpoint to check model status
@app.route('/model_status')
def model_status():
    return jsonify({"loaded": model is not None})

def cleanup():
    release_camera()
    hands.close()  # Clean up MediaPipe resources

if __name__ == '__main__':
    # Load the model first
    if not load_gesture_model():
        print("WARNING: Failed to load model. Continuing without model...")
        
    # Start the Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()