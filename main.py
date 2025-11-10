from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import tempfile
from typing import Dict, List, Optional
import uvicorn
import mediapipe as mp
from scipy.spatial import distance as dist

app = FastAPI(
    title="AI Video Annotator Service",
    description="Automated video annotation for eye state and posture detection",
    version="1.0.0"
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("MediaPipe Face Mesh initialized")

# MediaPipe eye landmarks (6 points per eye for EAR calculation)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# OpenCV Haar Cascade (backup method)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) - proven method for eye state detection.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical distances
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    if C > 0:
        ear = (A + B) / (2.0 * C)
    else:
        ear = 0.25  # Default to open
    
    return ear


def weighted_majority(buffer: List[str], weights: Optional[List[float]] = None) -> str:
    """
    Weighted voting for temporal smoothing - recent frames have higher weight.
    """
    if not buffer:
        return "Open"  # Default
    
    if weights is None:
        # Default weights: recent frames matter more
        weights = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6]
    
    weighted_votes = {}
    for i, value in enumerate(buffer):
        weight = weights[min(i, len(weights) - 1)]
        weighted_votes[value] = weighted_votes.get(value, 0) + weight
    
    return max(weighted_votes, key=weighted_votes.get)


def detect_eye_state_advanced(frame, frame_brightness: float = None) -> tuple:
    """
    Advanced eye detection using MediaPipe + OpenCV ensemble.
    Multi-method approach for robustness.
    
    Returns: (eye_state: str, confidence: float)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate frame brightness for adaptive thresholding
    if frame_brightness is None:
        frame_brightness = np.mean(gray)
    
    eye_scores = []
    confidences = []
    
    # Method 1: MediaPipe Face Mesh (PRIMARY - Most Accurate)
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Check landmark visibility/confidence
            nose_tip = face_landmarks.landmark[1]
            if nose_tip.visibility > 0.5 if hasattr(nose_tip, 'visibility') else True:
                # Extract left eye landmarks
                left_eye = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) 
                           for i in LEFT_EYE_INDICES]
                
                # Extract right eye landmarks
                right_eye = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) 
                            for i in RIGHT_EYE_INDICES]
                
                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                eye_scores.append(avg_ear)
                confidences.append(0.9)  # High confidence for MediaPipe
    except Exception as e:
        pass
    
    # Method 2: OpenCV Haar Cascade (BACKUP)
    try:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Use largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            (x, y, fw, fh) = face
            roi_gray = gray[y:y+fh, x:x+fw]
            
            # Eye region (top 20-60% of face)
            eye_region = roi_gray[int(fh*0.2):int(fh*0.6), :]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            
            # Adaptive brightness threshold
            adaptive_threshold = max(30, frame_brightness * 0.3)
            
            if len(eyes) >= 2:
                # Both eyes detected = definitely open
                eye_scores.append(0.28)
                confidences.append(0.7)
            elif len(eyes) == 1:
                # One eye - check brightness with adaptive threshold
                ex, ey, ew, eh = eyes[0]
                eye_patch = eye_region[ey:ey+eh, ex:ex+ew]
                if eye_patch.size > 0:
                    brightness = np.mean(eye_patch)
                    variance = np.var(eye_patch)
                    # Bright and high variance = open (adaptive threshold)
                    if variance > 200 and brightness > adaptive_threshold:
                        eye_scores.append(0.26)
                        confidences.append(0.6)
                    else:
                        eye_scores.append(0.18)
                        confidences.append(0.5)
            else:
                # No eyes detected - check region characteristics
                if eye_region.size > 0:
                    brightness = np.mean(eye_region)
                    variance = np.var(eye_region)
                    # Dark and low variance = closed
                    if brightness < adaptive_threshold and variance < 180:
                        eye_scores.append(0.16)
                        confidences.append(0.4)
                    else:
                        eye_scores.append(0.22)
                        confidences.append(0.4)
    except Exception as e:
        pass
    
    # Decision fusion with confidence weighting
    if len(eye_scores) == 0:
        return "Open", 0.3  # Low confidence default
    
    # Weighted average by confidence
    if len(confidences) == len(eye_scores):
        avg_ear = np.average(eye_scores, weights=confidences)
        avg_confidence = np.mean(confidences)
    else:
        avg_ear = np.mean(eye_scores)
        avg_confidence = 0.6
    
    # IMPROVED: Wider threshold gap for better separation
    CLOSED_THRESHOLD = 0.18  # Lowered from 0.20
    OPEN_THRESHOLD = 0.25    # Raised from 0.23
    
    if avg_ear < CLOSED_THRESHOLD:
        return "Closed", avg_confidence
    elif avg_ear > OPEN_THRESHOLD:
        return "Open", avg_confidence
    else:
        # Ambiguous zone - use confidence-weighted decision
        midpoint = (CLOSED_THRESHOLD + OPEN_THRESHOLD) / 2
        if avg_ear >= midpoint:
            return "Open", avg_confidence * 0.8
        else:
            return "Closed", avg_confidence * 0.8


def detect_posture_advanced(frame) -> tuple:
    """
    Advanced posture detection using MediaPipe + OpenCV multi-factor analysis.
    Ensemble scoring from multiple geometric features.
    
    Returns: (posture: str, confidence: float)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    posture_votes = []
    confidences = []
    
    # Method 1: MediaPipe Face Mesh (PRIMARY)
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Key facial points
            nose_tip = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            # Convert to pixel coordinates
            nose_y = nose_tip.y * h
            forehead_y = forehead.y * h
            chin_y = chin.y * h
            
            # Calculate metrics
            face_height = abs(chin_y - forehead_y)
            relative_nose_y = nose_y / h
            face_span_ratio = face_height / h
            
            # Multi-factor scoring (IMPROVED: Higher threshold)
            score = 0
            
            # Factor 1: Vertical position (lower = hunched)
            if relative_nose_y > 0.65:
                score += 5
            elif relative_nose_y > 0.58:
                score += 3
            elif relative_nose_y > 0.52:
                score += 1
            
            # Factor 2: Face size (smaller = farther/hunched)
            if face_span_ratio < 0.25:
                score += 5
            elif face_span_ratio < 0.32:
                score += 3
            elif face_span_ratio < 0.40:
                score += 1
            
            # Factor 3: Overall face span
            all_y_coords = [lm.y * h for lm in face_landmarks.landmark]
            total_span = (max(all_y_coords) - min(all_y_coords)) / h
            
            if total_span < 0.28:
                score += 3
            elif total_span < 0.36:
                score += 1
            
            # Factor 4: Face width analysis
            all_x_coords = [lm.x * w for lm in face_landmarks.landmark]
            face_width = (max(all_x_coords) - min(all_x_coords)) / w
            
            if face_width < 0.15:
                score += 2
            elif face_width < 0.20:
                score += 1
            
            # IMPROVED: Raised threshold from 6 to 8
            posture_votes.append("Hunched" if score >= 8 else "Straight")
            confidences.append(0.9)
    except Exception as e:
        pass
    
    # Method 2: OpenCV Haar Cascade (BACKUP)
    try:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            face = max(faces, key=lambda f: f[2] * f[3])
            (x, y, fw, fh) = face
            
            # Calculate metrics
            face_center_y = (y + fh/2) / h
            face_area_ratio = (fw * fh) / (w * h)
            aspect_ratio = fh / fw if fw > 0 else 1.0
            width_ratio = fw / w
            
            score = 0
            
            # Vertical position
            if face_center_y > 0.60:
                score += 4
            elif face_center_y > 0.52:
                score += 2
            
            # Face size
            if face_area_ratio < 0.07:
                score += 4
            elif face_area_ratio < 0.11:
                score += 2
            elif face_area_ratio < 0.15:
                score += 1
            
            # Aspect ratio (wider than tall = hunched)
            if aspect_ratio < 1.15:
                score += 3
            elif aspect_ratio < 1.25:
                score += 1
            
            # Width
            if width_ratio < 0.18:
                score += 1
            
            # IMPROVED: Raised threshold from 5 to 7
            posture_votes.append("Hunched" if score >= 7 else "Straight")
            confidences.append(0.7)
    except Exception as e:
        pass
    
    # Ensemble decision with confidence
    if len(posture_votes) == 0:
        return "Straight", 0.3  # Low confidence default
    
    # Confidence-weighted voting
    weighted_votes = {}
    for vote, conf in zip(posture_votes, confidences):
        weighted_votes[vote] = weighted_votes.get(vote, 0) + conf
    
    result = max(weighted_votes, key=weighted_votes.get)
    avg_confidence = np.mean(confidences)
    
    return result, avg_confidence


def process_video(video_path: str) -> Dict:
    """
    Process video with advanced hybrid detection and improved temporal smoothing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    labels_per_frame = {}
    
    print(f"\n{'='*60}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Frames: {total_frames} @ {fps:.1f} FPS")
    print(f"AI Models: MediaPipe Face Mesh + OpenCV")
    print(f"{'='*60}\n")
    
    # Temporal smoothing buffers with improved weighting
    eye_buffer = []
    posture_buffer = []
    BUFFER_SIZE = 7  # Larger buffer for more stability
    
    # Weights for temporal smoothing (recent frames have more influence)
    temporal_weights = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6]
    
    frame_idx = 0
    prev_label = {"eye_state": "Open", "posture": "Straight"}
    
    # Calculate global frame brightness for adaptive processing
    frame_brightnesses = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Calculate frame brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_brightness = np.mean(gray)
            frame_brightnesses.append(frame_brightness)
            
            # Use average brightness for adaptive thresholding
            avg_brightness = np.mean(frame_brightnesses[-30:])  # Last 1 second
            
            # Process frame with confidence scores
            eye_state, eye_conf = detect_eye_state_advanced(frame, avg_brightness)
            posture, posture_conf = detect_posture_advanced(frame)
            
            # Add to temporal buffers
            eye_buffer.append(eye_state)
            posture_buffer.append(posture)
            
            if len(eye_buffer) > BUFFER_SIZE:
                eye_buffer.pop(0)
            if len(posture_buffer) > BUFFER_SIZE:
                posture_buffer.pop(0)
            
            # IMPROVED: Weighted voting for temporal consistency
            final_eye = weighted_majority(eye_buffer, temporal_weights)
            final_posture = weighted_majority(posture_buffer, temporal_weights)
            
            labels_per_frame[str(frame_idx)] = {
                "eye_state": final_eye,
                "posture": final_posture
            }
            
            prev_label = labels_per_frame[str(frame_idx)]
            
        except Exception as e:
            print(f"Frame {frame_idx} error: {str(e)}")
            # Use previous frame (more graceful degradation)
            labels_per_frame[str(frame_idx)] = prev_label.copy()
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            progress = int(frame_idx / total_frames * 100)
            print(f"{frame_idx}/{total_frames} ({progress}%)")
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"Complete: {frame_idx} frames annotated")
    print(f"{'='*60}\n")
    
    return {
        "video_filename": os.path.basename(video_path),
        "total_frames": frame_idx,
        "labels_per_frame": labels_per_frame
    }


@app.post("/annotate")
async def annotate_video(file: UploadFile = File(...)):
    """
    Annotate video with eye state and posture labels.
    
    Accepts .mp4 or .avi video files and returns frame-by-frame analysis.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi')):
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 and .avi formats supported"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        file_size_mb = len(content) / (1024 * 1024)
        print(f"\n Received: {file.filename} ({file_size_mb:.2f} MB)")
        
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
        "version": "1.0.0",
        "status": "running",
        "models": "MediaPipe Face Mesh + OpenCV",
        "improvements": "Enhanced thresholds, weighted temporal smoothing, adaptive lighting",
        "endpoints": {
            "annotate": "/annotate (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mediapipe": "loaded",
        "opencv": "loaded"
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI Video Annotator Service - Enhanced Accuracy")
    print("="*60)
    print("Using MediaPipe Face Mesh + OpenCV")
    print("Improvements:")
    print("   - Wider EAR thresholds (0.18/0.25)")
    print("   - Raised posture detection thresholds")
    print("   - Weighted temporal smoothing")
    print("   - Adaptive lighting compensation")
    print("   - Confidence-based decision fusion")
    print("Server: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")