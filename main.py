from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import tempfile
from typing import Dict, Tuple
import uvicorn
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import warnings

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
warnings.filterwarnings('ignore')

app = FastAPI(
    title="AI Video Annotator Service",
    description="Video annotation for eye state and posture detection",
    version="13.0.0"
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

EYE_AR_THRESHOLD = 0.18


def calculate_ear(eye_landmarks) -> float:
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    if C < 0.001:
        return 0.25
    
    return (A + B) / (2.0 * C)


def calculate_angle_3d(p1: Tuple[float, float, float], 
                       p2: Tuple[float, float, float], 
                       p3: Tuple[float, float, float]) -> float:
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def detect_eye_state(frame) -> Tuple[str, float]:
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return "Open", 0.0
    
    face_landmarks = results.multi_face_landmarks[0]
    
    left_eye_points = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) 
                       for i in LEFT_EYE_INDICES]
    right_eye_points = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) 
                        for i in RIGHT_EYE_INDICES]
    
    left_ear = calculate_ear(left_eye_points)
    right_ear = calculate_ear(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2.0
    
    threshold_low = 0.16
    threshold_high = 0.20
    
    if avg_ear < threshold_low:
        return "Closed", 0.92
    elif avg_ear < threshold_high:
        return "Closed", 0.75
    else:
        return "Open", 0.92


def detect_posture(frame) -> Tuple[str, float]:
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pose_results = pose.process(frame_rgb)
    
    if not pose_results.pose_landmarks:
        return "Hunched", 0.0
    
    landmarks = pose_results.pose_landmarks.landmark
    
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        
        visibility_threshold = 0.3
        if not all([
            left_shoulder.visibility > visibility_threshold,
            right_shoulder.visibility > visibility_threshold,
            left_hip.visibility > visibility_threshold,
            right_hip.visibility > visibility_threshold
        ]):
            return "Hunched", 0.0
        
        shoulder_mid = (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        )
        
        hip_mid = (
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        )
        
        ear_mid = (
            (left_ear.x + right_ear.x) / 2,
            (left_ear.y + right_ear.y) / 2,
            (left_ear.z + right_ear.z) / 2
        )
        
        torso_vector = np.array([
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1],
            shoulder_mid[2] - hip_mid[2]
        ])
        
        torso_length = np.linalg.norm(torso_vector)
        
        if torso_length < 0.05:
            return "Hunched", 0.0
        
        vertical_distance = shoulder_mid[1] - ear_mid[1]
        normalized_vertical = vertical_distance / torso_length
        
        neck_angle = calculate_angle_3d(ear_mid, shoulder_mid, hip_mid)
        
        shoulder_forward = shoulder_mid[2] - hip_mid[2]
        normalized_shoulder_forward = shoulder_forward / torso_length
        
        ear_forward = ear_mid[2] - shoulder_mid[2]
        normalized_ear_forward = ear_forward / torso_length
        
        is_head_high = normalized_vertical > 0.80
        is_neck_straight = neck_angle > 165
        is_body_upright = normalized_shoulder_forward < 0.12
        is_head_back = normalized_ear_forward < 0.10
        
        straight_score = sum([is_head_high, is_neck_straight, is_body_upright, is_head_back])
        
        if straight_score >= 3:
            return "Straight", 0.89
        else:
            return "Hunched", 0.89
            
    except Exception:
        return "Hunched", 0.0


class TemporalSmoother:
    def __init__(self, posture_buffer_size=11):
        self.posture_buffer_size = posture_buffer_size
        self.posture_buffer = deque(maxlen=posture_buffer_size)
    
    def add_posture(self, posture):
        self.posture_buffer.append(posture)
    
    def get_smoothed_posture(self) -> str:
        if not self.posture_buffer:
            return "Hunched"
        
        if len(self.posture_buffer) < 3:
            return self.posture_buffer[-1]
        
        votes = {"Straight": 0, "Hunched": 0}
        for i, state in enumerate(self.posture_buffer):
            weight = (i + 1) ** 1.3
            votes[state] += weight
        
        return max(votes, key=votes.get)


def process_video(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    labels_per_frame = {}
    
    print(f"\nProcessing: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames} at {fps:.1f} FPS\n")
    
    smoother = TemporalSmoother(posture_buffer_size=11)
    
    frame_idx = 0
    last_valid_eye = "Open"
    last_valid_posture = "Hunched"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            eye_state, eye_conf = detect_eye_state(frame)
            posture, posture_conf = detect_posture(frame)
            
            if eye_conf < 0.3:
                eye_state = last_valid_eye
            else:
                last_valid_eye = eye_state
            
            if posture_conf < 0.3:
                posture = last_valid_posture
            else:
                last_valid_posture = posture
            
            smoother.add_posture(posture)
            
            final_posture = smoother.get_smoothed_posture()
            
            labels_per_frame[str(frame_idx)] = {
                "eye_state": eye_state,
                "posture": final_posture
            }
            
        except Exception:
            labels_per_frame[str(frame_idx)] = {
                "eye_state": last_valid_eye,
                "posture": last_valid_posture
            }
        
        frame_idx += 1
        
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            progress = int(frame_idx / total_frames * 100)
            print(f"Progress: {frame_idx}/{total_frames} ({progress}%)")
    
    cap.release()
    
    print(f"\nProcessing complete: {frame_idx} frames processed\n")
    
    return {
        "video_filename": os.path.basename(video_path),
        "total_frames": frame_idx,
        "labels_per_frame": labels_per_frame
    }


@app.post("/annotate")
async def annotate_video(file: UploadFile = File(...)):
    allowed_extensions = (
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
        '.webm', '.m4v', '.mpeg', '.mpg', '.3gp', '.ogv'
    )
    
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        print(f"\nReceived: {file.filename} ({len(content) / (1024 * 1024):.2f} MB)")
        
        result = process_video(tmp_file_path)
        result["video_filename"] = file.filename
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.get("/")
async def root():
    return {
        "service": "AI Video Annotator",
        "version": "13.0.0",
        "status": "running",
        "endpoints": {
            "annotate": "/annotate (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mediapipe_face": "loaded",
        "mediapipe_pose": "loaded"
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI Video Annotator Service v13.0 - HUNCHED DEFAULT")
    print("="*60)
    print("Server: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")