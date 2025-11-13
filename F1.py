import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import sys


def load_json_safe(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"\nERROR: File '{filepath}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\nERROR: Invalid JSON in '{filepath}'")
        sys.exit(1)


def extract_labels(predictions, ground_truth):
    pred_labels = predictions.get('labels_per_frame', {})
    
    if 'manual_labels_per_frame' in ground_truth:
        gt_labels = ground_truth['manual_labels_per_frame']
    elif 'labels_per_frame' in ground_truth:
        gt_labels = ground_truth['labels_per_frame']
    else:
        print("\nERROR: Cannot find labels in ground truth.")
        sys.exit(1)

    eye_gt = []
    eye_pred = []
    posture_gt = []
    posture_pred = []

    eye_map = {'Open': 0, 'Closed': 1}
    posture_map = {'Straight': 0, 'Hunched': 1}

    gt_frame_keys = sorted(gt_labels.keys(), key=int)
    pred_frame_keys = sorted(pred_labels.keys(), key=int)
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS - CHECKING ACTUAL VALUES IN FILES")
    print(f"{'='*70}")
    
    pred_posture_counts = {'Straight': 0, 'Hunched': 0, 'Other': 0}
    gt_posture_counts = {'Straight': 0, 'Hunched': 0, 'Other': 0}
    
    for frame_num in pred_frame_keys[:10]:
        p_data = pred_labels[frame_num]
        print(f"Prediction Frame {frame_num}: {p_data}")
        
    for frame_num in gt_frame_keys[:10]:
        g_data = gt_labels[frame_num]
        print(f"Ground Truth Frame {frame_num}: {g_data}")
    
    skipped_frames = 0
    
    for frame_num in pred_frame_keys:
        if frame_num not in gt_labels:
            skipped_frames += 1
            continue
        
        try:
            p_eye = pred_labels[frame_num].get('eye_state')
            p_posture = pred_labels[frame_num].get('posture')
            
            gt_frame = gt_labels[frame_num]
            g_eye = gt_frame.get('open_closed')
            g_hunched_bool = gt_frame.get('hunched')
            
            if g_hunched_bool is True:
                g_posture = 'Hunched'
            elif g_hunched_bool is False:
                g_posture = 'Straight'
            else:
                g_posture = None
            
            if p_posture in posture_map:
                pred_posture_counts[p_posture] += 1
            else:
                pred_posture_counts['Other'] += 1
                
            if g_posture and g_posture in posture_map:
                gt_posture_counts[g_posture] += 1
            else:
                gt_posture_counts['Other'] += 1
            
            if p_eye is None or g_eye is None or p_posture is None or g_posture is None:
                skipped_frames += 1
                continue

            if g_eye in eye_map and p_eye in eye_map:
                eye_gt.append(eye_map[g_eye])
                eye_pred.append(eye_map[p_eye])
            else:
                skipped_frames += 1
                continue

            if g_posture in posture_map and p_posture in posture_map:
                posture_gt.append(posture_map[g_posture])
                posture_pred.append(posture_map[p_posture])
            else:
                skipped_frames += 1
                continue

        except (KeyError, TypeError) as e:
            skipped_frames += 1
            continue

    print(f"\n{'='*70}")
    print("ACTUAL COUNTS IN FILES:")
    print(f"{'='*70}")
    print(f"Ground Truth Posture: {gt_posture_counts}")
    print(f"Predictions Posture: {pred_posture_counts}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Frames with valid eye data: {len(eye_gt)}")
    print(f"Frames with valid posture data: {len(posture_gt)}")
    
    gt_eye_counts = {'Open': 0, 'Closed': 0}
    pred_eye_counts = {'Open': 0, 'Closed': 0}
    for i in range(len(eye_gt)):
        if eye_gt[i] == 0:
            gt_eye_counts['Open'] += 1
        else:
            gt_eye_counts['Closed'] += 1
        if eye_pred[i] == 0:
            pred_eye_counts['Open'] += 1
        else:
            pred_eye_counts['Closed'] += 1
    
    print(f"\nGround Truth Eye State: {gt_eye_counts}")
    print(f"Predictions Eye State: {pred_eye_counts}")
    print(f"{'='*70}\n")

    if not eye_gt or not posture_gt:
        print("\nERROR: Insufficient valid frames.")
        sys.exit(1)

    return {
        'eye_gt': np.array(eye_gt),
        'eye_pred': np.array(eye_pred),
        'posture_gt': np.array(posture_gt),
        'posture_pred': np.array(posture_pred),
        'total_frames_evaluated': len(eye_gt),
        'total_frames_in_gt': len(gt_frame_keys),
        'total_frames_in_pred': len(pred_frame_keys)
    }


def calculate_metrics(y_true, y_pred, label_names):
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    class_metrics = {}
    for idx, label in enumerate(label_names):
        if idx < len(per_class_f1):
            class_metrics[label] = {
                'f1': float(per_class_f1[idx]),
                'precision': float(per_class_precision[idx]),
                'recall': float(per_class_recall[idx])
            }
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'per_class': class_metrics
    }


def calculate_f1_scores(prediction_filepath, ground_truth_filepath):
    predictions = load_json_safe(prediction_filepath)
    ground_truth = load_json_safe(ground_truth_filepath)
    
    data = extract_labels(predictions, ground_truth)
    
    eye_metrics = calculate_metrics(data['eye_gt'], data['eye_pred'], ['Open', 'Closed'])
    posture_metrics = calculate_metrics(data['posture_gt'], data['posture_pred'], ['Straight', 'Hunched'])
    
    return {
        'eye_state': {
            'f1': float(eye_metrics['f1']),
            'precision': float(eye_metrics['precision']),
            'recall': float(eye_metrics['recall']),
            'accuracy': float(eye_metrics['accuracy']),
            'confusion_matrix': eye_metrics['confusion_matrix'],
            'per_class': eye_metrics['per_class']
        },
        'posture': {
            'f1': float(posture_metrics['f1']),
            'precision': float(posture_metrics['precision']),
            'recall': float(posture_metrics['recall']),
            'accuracy': float(posture_metrics['accuracy']),
            'confusion_matrix': posture_metrics['confusion_matrix'],
            'per_class': posture_metrics['per_class']
        },
        'overall': {
            'average_f1': float((eye_metrics['f1'] + posture_metrics['f1']) / 2),
            'frames_evaluated': data['total_frames_evaluated'],
            'frames_in_ground_truth': data['total_frames_in_gt'],
            'frames_in_predictions': data['total_frames_in_pred']
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate F1 scores for video annotation')
    parser.add_argument('--pred', default='result.json', help='Prediction JSON file')
    parser.add_argument('--gt', default='ground_truth.json', help='Ground truth JSON file')
    
    args = parser.parse_args()
    
    try:
        results = calculate_f1_scores(args.pred, args.gt)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nEye State Metrics:")
        print(f"  F1-Score:   {results['eye_state']['f1']:.4f}")
        print(f"  Precision:  {results['eye_state']['precision']:.4f}")
        print(f"  Recall:     {results['eye_state']['recall']:.4f}")
        print(f"  Accuracy:   {results['eye_state']['accuracy']:.4f}")
        
        print(f"\nPosture Metrics:")
        print(f"  F1-Score:   {results['posture']['f1']:.4f}")
        print(f"  Precision:  {results['posture']['precision']:.4f}")
        print(f"  Recall:     {results['posture']['recall']:.4f}")
        print(f"  Accuracy:   {results['posture']['accuracy']:.4f}")
        
        print(f"\nOverall:")
        print(f"  Average F1: {results['overall']['average_f1']:.4f}")
        print(f"  Frames Evaluated: {results['overall']['frames_evaluated']}")
        
        print("\n" + "="*70)
        
    except SystemExit:
        pass
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()