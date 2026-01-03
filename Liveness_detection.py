import cv2
import numpy as np
import mediapipe as mp
from scipy import signal
from collections import deque
import time

class DeepfakeDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe eye landmarks indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Eye blink detection
        self.EYE_AR_THRESH = 0.21
        self.EYE_AR_CONSEC_FRAMES = 3
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_history = deque(maxlen=100)
        
        # rPPG (remote photoplethysmography) for heart rate
        self.rppg_buffer = deque(maxlen=300)  # 10 seconds at 30 fps
        self.heart_rate = 0
        self.heart_rate_history = deque(maxlen=50)
        
        # Frequency/Texture analysis
        self.texture_history = deque(maxlen=30)
        self.frequency_scores = deque(maxlen=30)
        
        # Motion consistency
        self.motion_history = deque(maxlen=30)
        self.prev_gray = None
        
        # Head pose stability
        self.head_pose_history = deque(maxlen=30)
        
        # Overall detection scores
        self.detection_scores = {
            'blink': 0,
            'rppg': 0,
            'texture': 0,
            'motion': 0,
            'pose': 0
        }
        
    def eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio for blink detection"""
        # Compute the euclidean distances between vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_points(self, landmarks, eye_indices, frame_shape):
        """Extract eye points from MediaPipe landmarks"""
        h, w = frame_shape[:2]
        points = []
        for idx in eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        return np.array(points, dtype=np.float32)
    
    def detect_blinks(self, landmarks, frame_shape):
        """Detect eye blinks using Eye Aspect Ratio"""
        left_eye = self.get_eye_points(landmarks, self.LEFT_EYE, frame_shape)
        right_eye = self.get_eye_points(landmarks, self.RIGHT_EYE, frame_shape)
        
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.EYE_AR_THRESH:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
                self.blink_history.append(time.time())
            self.blink_counter = 0
        
        # Calculate blink rate (blinks per minute)
        if len(self.blink_history) > 1:
            time_span = self.blink_history[-1] - self.blink_history[0]
            if time_span > 0:
                blink_rate = len(self.blink_history) / time_span * 60
                # Normal blink rate: 15-20 per minute
                if 10 < blink_rate < 30:
                    self.detection_scores['blink'] = min(100, self.detection_scores['blink'] + 2)
                else:
                    self.detection_scores['blink'] = max(0, self.detection_scores['blink'] - 5)
        
        return ear
    
    def get_face_region(self, landmarks, frame):
        """Extract forehead region for rPPG analysis"""
        h, w = frame.shape[:2]
        
        # Get forehead landmarks (top of face)
        forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                           397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                           172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        points = []
        for idx in forehead_indices:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                points.append([int(landmark.x * w), int(landmark.y * h)])
        
        if len(points) == 0:
            return None
        
        points = np.array(points)
        
        # Create bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Extract region
        roi = frame[max(0, y_min):min(h, y_max), max(0, x_min):min(w, x_max)]
        return roi
    
    def extract_rppg_signal(self, frame, landmarks):
        """Extract rPPG signal from facial region"""
        forehead = self.get_face_region(landmarks, frame)
        
        if forehead is None or forehead.size == 0:
            return
        
        # Calculate mean color values
        mean_rgb = np.mean(forehead, axis=(0, 1))
        self.rppg_buffer.append(mean_rgb)
        
        if len(self.rppg_buffer) >= 150:  # Need at least 5 seconds
            # Convert to numpy array and detrend
            signal_array = np.array(self.rppg_buffer)
            
            # Use green channel (most sensitive to blood volume changes)
            green_signal = signal_array[:, 1]
            detrended = signal.detrend(green_signal)
            
            # Apply bandpass filter (0.7-4 Hz for 42-240 BPM)
            fps = 30
            nyquist = fps / 2
            low = 0.7 / nyquist
            high = 4.0 / nyquist
            
            b, a = signal.butter(3, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
            
            # Perform FFT
            fft_result = np.fft.fft(filtered)
            freqs = np.fft.fftfreq(len(filtered), 1/fps)
            
            # Find dominant frequency in valid range
            valid_idx = np.where((freqs > 0.7) & (freqs < 4.0))
            if len(valid_idx[0]) > 0:
                valid_fft = np.abs(fft_result[valid_idx])
                valid_freqs = freqs[valid_idx]
                
                dominant_freq = valid_freqs[np.argmax(valid_fft)]
                self.heart_rate = dominant_freq * 60  # Convert to BPM
                self.heart_rate_history.append(self.heart_rate)
                
                # Check if heart rate is in realistic range (50-120 BPM)
                if 50 <= self.heart_rate <= 120:
                    # Check for consistency
                    if len(self.heart_rate_history) > 10:
                        hr_std = np.std(list(self.heart_rate_history)[-10:])
                        if hr_std < 15:  # Realistic variation
                            self.detection_scores['rppg'] = min(100, self.detection_scores['rppg'] + 3)
                        else:
                            self.detection_scores['rppg'] = max(0, self.detection_scores['rppg'] - 2)
                else:
                    self.detection_scores['rppg'] = max(0, self.detection_scores['rppg'] - 5)
    
    def get_face_bbox(self, landmarks, frame_shape):
        """Get bounding box of face"""
        h, w = frame_shape[:2]
        
        x_coords = [landmark.x * w for landmark in landmarks]
        y_coords = [landmark.y * h for landmark in landmarks]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def analyze_texture_artifacts(self, frame, landmarks):
        """Detect texture artifacts common in deepfakes"""
        x, y, w, h = self.get_face_bbox(landmarks, frame.shape)
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Compute high-frequency components
        dft = cv2.dft(np.float32(gray_face), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        # Check high-frequency energy
        rows, cols = gray_face.shape
        crow, ccol = rows // 2, cols // 2
        high_freq_region = magnitude[0:crow//2, :].sum() + magnitude[crow+crow//2:, :].sum()
        total_energy = magnitude.sum()
        high_freq_ratio = high_freq_region / (total_energy + 1e-6)
        
        self.texture_history.append(laplacian_var)
        self.frequency_scores.append(high_freq_ratio)
        
        # Deepfakes often have smoother textures and different frequency distributions
        if len(self.texture_history) > 10:
            texture_std = np.std(list(self.texture_history)[-10:])
            freq_mean = np.mean(list(self.frequency_scores)[-10:])
            
            # Real faces have more texture variation
            if texture_std > 50 and 0.2 < freq_mean < 0.5:
                self.detection_scores['texture'] = min(100, self.detection_scores['texture'] + 2)
            else:
                self.detection_scores['texture'] = max(0, self.detection_scores['texture'] - 3)
    
    def analyze_motion_consistency(self, frame):
        """Check for motion consistency issues common in deepfakes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(mag)
            
            self.motion_history.append(motion_score)
            
            # Check for unnatural motion patterns
            if len(self.motion_history) > 10:
                motion_std = np.std(list(self.motion_history)[-10:])
                motion_mean = np.mean(list(self.motion_history)[-10:])
                
                # Real motion should have natural variation
                if 0.5 < motion_std < 5 and motion_mean > 0.1:
                    self.detection_scores['motion'] = min(100, self.detection_scores['motion'] + 2)
                else:
                    self.detection_scores['motion'] = max(0, self.detection_scores['motion'] - 2)
        
        self.prev_gray = gray.copy()
    
    def analyze_head_pose(self, landmarks, frame_shape):
        """Analyze head pose stability (deepfakes often have unnatural head movements)"""
        h, w = frame_shape[:2]
        
        # Get key points for pose estimation
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calculate angles
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        
        # Calculate head tilt
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        angle = np.degrees(np.arctan2(dy, dx))
        
        self.head_pose_history.append(angle)
        
        if len(self.head_pose_history) > 10:
            pose_std = np.std(list(self.head_pose_history)[-10:])
            
            # Natural head movements have some variation but not too erratic
            if 2 < pose_std < 15:
                self.detection_scores['pose'] = min(100, self.detection_scores['pose'] + 2)
            else:
                self.detection_scores['pose'] = max(0, self.detection_scores['pose'] - 2)
    
    def get_overall_score(self):
        """Calculate overall authenticity score"""
        weights = {
            'blink': 0.20,
            'rppg': 0.30,
            'texture': 0.25,
            'motion': 0.15,
            'pose': 0.10
        }
        
        total_score = sum(self.detection_scores[k] * weights[k] for k in weights)
        return total_score
    
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        detection_results = {
            'face_detected': False,
            'ear': 0,
            'blinks': self.total_blinks,
            'heart_rate': self.heart_rate,
            'overall_score': self.get_overall_score(),
            'scores': self.detection_scores.copy(),
            'verdict': 'ANALYZING...'
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            detection_results['face_detected'] = True
            
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Perform all detections
            ear = self.detect_blinks(face_landmarks.landmark, frame.shape)
            detection_results['ear'] = ear
            
            self.extract_rppg_signal(frame, face_landmarks.landmark)
            self.analyze_texture_artifacts(frame, face_landmarks.landmark)
            self.analyze_motion_consistency(frame)
            self.analyze_head_pose(face_landmarks.landmark, frame.shape)
            
            # Update overall score and verdict
            overall_score = self.get_overall_score()
            detection_results['overall_score'] = overall_score
            detection_results['scores'] = self.detection_scores.copy()
            
            if overall_score > 70:
                detection_results['verdict'] = 'LIKELY REAL'
                verdict_color = (0, 255, 0)
            elif overall_score > 40:
                detection_results['verdict'] = 'UNCERTAIN'
                verdict_color = (0, 165, 255)
            else:
                detection_results['verdict'] = 'POSSIBLE DEEPFAKE'
                verdict_color = (0, 0, 255)
            
            # Draw bounding box
            x, y, w, h = self.get_face_bbox(face_landmarks.landmark, frame.shape)
            cv2.rectangle(frame, (x, y), (x+w, y+h), verdict_color, 2)
        
        return frame, detection_results

def main():
    """Main function to run the deepfake detector"""
    print("Initializing Real-time Deepfake Detection with MediaPipe...")
    detector = DeepfakeDetector()
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    print("Starting detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        processed_frame, results = detector.process_frame(frame)
        
        # Display results on frame
        y_offset = 30
        
        # Verdict with color
        if results['verdict'] == 'LIKELY REAL':
            color = (0, 255, 0)
        elif results['verdict'] == 'UNCERTAIN':
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
            
        cv2.putText(processed_frame, f"Verdict: {results['verdict']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
        
        cv2.putText(processed_frame, f"Overall Score: {results['overall_score']:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        cv2.putText(processed_frame, f"Blinks: {results['blinks']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(processed_frame, f"Heart Rate: {results['heart_rate']:.1f} BPM", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(processed_frame, f"EAR: {results['ear']:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 35
        
        # Display individual scores with color coding
        for key, value in results['scores'].items():
            if value > 50:
                score_color = (0, 255, 0)
            elif value > 25:
                score_color = (0, 165, 255)
            else:
                score_color = (0, 0, 255)
                
            cv2.putText(processed_frame, f"{key}: {value:.0f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)
            y_offset += 20
        
        cv2.imshow('Deepfake Detection - MediaPipe', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()