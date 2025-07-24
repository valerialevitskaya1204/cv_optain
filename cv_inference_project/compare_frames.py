import json
import argparse
import numpy as np
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Compare consecutive frames in range')
    parser.add_argument('--summary', required=True, help='Path to summary.json')
    parser.add_argument('--frame1', type=int, required=True, help='Start frame index')
    parser.add_argument('--frame2', type=int, required=True, help='End frame index')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()

    with open(args.summary, 'r') as f:
        summary = json.load(f)

    # Initialize comparison structure
    comparison = {
        "frame_range": [args.frame1, args.frame2],
        "pairwise_comparisons": []
    }

    # Get all frames in range for each model
    model_frames = {
        model: sorted([f for f in frames if args.frame1 <= f['frame'] <= args.frame2], 
                     key=lambda x: x['frame'])
        for model, frames in summary.items()
    }

    # Get all unique frame numbers in range
    all_frames = sorted(set(
        f['frame'] 
        for frames in model_frames.values() 
        for f in frames
    ))

    # Process each consecutive pair
    for i in range(1, len(all_frames)):
        frame_prev = all_frames[i-1]
        frame_current = all_frames[i]
        
        pair_result = {
            "frame1": frame_prev,
            "frame2": frame_current,
            "results": {}
        }

        for model, frames in model_frames.items():
            # Find the specific frames we want to compare
            prev_data = next((f['meta'] for f in frames if f['frame'] == frame_prev), None)
            curr_data = next((f['meta'] for f in frames if f['frame'] == frame_current), None)
            
            if not prev_data or not curr_data:
                continue
                
            if model == "gaze":
                angle_diff = abs(prev_data['gaze_angle'] - curr_data['gaze_angle'])
                flag_changed = prev_data['gaze_away'] != curr_data['gaze_away']
                pair_result["results"]["gaze"] = {
                    "angle_diff": angle_diff,
                    "flag_changed": flag_changed
                }
                
            elif model == "headpose":
                yaw_diff = abs(prev_data['yaw'] - curr_data['yaw'])
                pitch_diff = abs(prev_data['pitch'] - curr_data['pitch'])
                roll_diff = abs(prev_data['roll'] - curr_data['roll'])
                pair_result["results"]["headpose"] = {
                    "yaw_diff": yaw_diff,
                    "pitch_diff": pitch_diff,
                    "roll_diff": roll_diff
                }
                
            elif model == "identity":
                dist_diff = abs(prev_data['distance'] - curr_data['distance'])
                match_changed = prev_data['is_match'] != curr_data['is_match']
                pair_result["results"]["identity"] = {
                    "distance_diff": dist_diff,
                    "match_changed": match_changed
                }
                
            elif model == "persons":
                count_diff = curr_data['person_count'] - prev_data['person_count']
                pair_result["results"]["persons"] = {
                    "count_diff": count_diff
                }
                
            elif model == "phone":
                count_diff = curr_data['phone_count'] - prev_data['phone_count']
                pair_result["results"]["phone"] = {
                    "count_diff": count_diff
                }

        comparison["pairwise_comparisons"].append(pair_result)

    # Calculate some aggregate statistics
    comparison["total_comparisons"] = len(comparison["pairwise_comparisons"])
    
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)

if __name__ == "__main__":
    main()