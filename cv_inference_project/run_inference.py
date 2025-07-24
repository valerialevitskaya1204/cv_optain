import os
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import sys
import json
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from importlib import import_module
import decord
import cv2
import numpy as np

VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi')

def setup_logger(out_dir, level=logging.DEBUG):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inference")
    logger.setLevel(level)
    
    fh = logging.FileHandler(out_dir / "run_inference.log", mode='w')
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_all_videos(dataset_root):
    dataset_root = Path(dataset_root)
    return [
        (student_dir.name, video_path)
        for student_dir in dataset_root.iterdir()
        if student_dir.is_dir()
        for video_path in student_dir.rglob("*")
        if video_path.suffix.lower() in VIDEO_EXTS
    ]

def save_summary(out_dir, summaries, video_name):
    """Save summary data to JSON file"""
    summary_path = out_dir / f"{video_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    return summary_path

def extract_and_run(models, video_path, out_dir, frame_skip, logger):
    try:
        ctx = decord.cpu(0)
        vr = decord.VideoReader(str(video_path), ctx=ctx)
    except Exception as e:
        logger.error(f"Failed to open video {video_path}: {str(e)}")
        return None

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    if total_frames == 0:
        logger.error("Video has 0 frames")
        return None
    
    frames_10min = int(6 * fps)
    end_frame = min(frames_10min, total_frames)
    
    frame_indices = range(0, end_frame, frame_skip)
    
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    summaries = {m: [] for m in models}

    for idx in tqdm(frame_indices, desc=f"Processing {video_path.name}"):
        try:
            frame = vr[idx].asnumpy()
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            for model_name, model in models.items():
                try:
                    result = model.predict(frame.copy())
                    out_img, meta = (result if isinstance(result, tuple) 
                                   else (result, {}))
                    
                    # Convert all non-serializable types in metadata
                    meta = convert_to_serializable(meta)
                    
                    frame_out_path = frame_dir / f"frame_{idx:05d}_{model_name}.jpg"
                    cv2.imwrite(str(frame_out_path), out_img)
                    
                    summaries[model_name].append({
                        "frame": idx,
                        "timestamp": idx / fps,
                        "meta": meta
                    })
                except Exception as e:
                    logger.error(f"[{model_name}] failed on frame {idx}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing frame {idx}: {str(e)}")
            continue

    # Convert all summaries to serializable format
    for model_name in summaries:
        summaries[model_name] = convert_to_serializable(summaries[model_name])
    
    summary_path = save_summary(out_dir, summaries, video_path.stem)
    logger.info(f"Saved summary to {summary_path}")
    return summaries

def convert_to_serializable(obj):
    """Recursively convert objects to JSON-serializable formats"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif hasattr(obj, 'numpy'):  # TensorFlow tensors
        return obj.numpy().tolist()
    elif hasattr(obj, 'detach'):  # PyTorch tensors
        return obj.detach().cpu().numpy().tolist()
    elif hasattr(obj, 'tolist'):  # Other tensor-like objects
        return obj.tolist()
    else:
        # Try to convert to string representation as a fallback
        try:
            return str(obj)
        except:
            return f"Unserializable object: {type(obj)}"

def load_models(model_names, logger):
    models = {}
    for name in model_names:
        try:
            module = import_module(f"models.{name}")
            model = module.load_model()
            logger.info(f"Loaded model: {name}")
            models[name] = model
        except Exception as e:
            logger.error(f"Failed to load model {name}: {str(e)}")
    return models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["identity", "gaze", "headpose", "phone", "persons"])
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logger = setup_logger(args.output_dir, getattr(logging, args.log_level.upper()))
    
    try:
        import decord
        logger.info(f"Using decord version {decord.__version__}")
    except Exception as e:
        logger.critical(f"Decord import failed: {str(e)}")
        sys.exit(1)

    videos = get_all_videos(args.dataset_root)
    logger.info(f"Found {len(videos)} videos")

    models = load_models(args.models, logger)
    if not models:
        logger.error("No models loaded")
        sys.exit(1)

    all_results = {}
    for student_id, video_path in videos:
        out_dir = Path(args.output_dir) / student_id / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {video_path} → {out_dir}")
        
        summaries = extract_and_run(models, video_path, out_dir, args.frame_skip, logger)
        if summaries:
            all_results[f"{student_id}/{video_path.name}"] = {
                "summary_path": str(out_dir / f"{video_path.stem}_summary.json"),
                "frame_count": len(next(iter(summaries.values()))),
                "models": list(summaries.keys())
            }

    # Convert all results to serializable format
    all_results = convert_to_serializable(all_results)
    
    master_results_path = Path(args.output_dir) / "all_results.json"
    with open(master_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved master results to {master_results_path}")

if __name__ == "__main__":
    main()