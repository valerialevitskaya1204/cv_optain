from openai import OpenAI
import json
import os
import math
from typing import Dict, Any


import json
from datetime import datetime

def save_prompt_to_json(prompt: str, filename: str = None) -> str:
    """
    Сохраняет промпт в JSON файл с метаданными
    
    Args:
        prompt: Текст промпта для сохранения
        filename: Имя файла (если None, будет сгенерировано автоматически)
    
    Returns:
        Путь к сохраненному файлу
    """
    prompt_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "source": "video_analysis_tool"
        },
        "prompt": prompt
    }

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompt_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, ensure_ascii=False, indent=2)
    
    return filename

#deepseekr1
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

def load_comparison_data(file_path: str) -> Dict[str, Any]:
    """Load frame comparison data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def safe_float(value: Any) -> float:
    """convert value to float, with Infinity and large numbers"""
    if isinstance(value, (int, float)):
        return float(value)
    elif value == "Infinity" or value == "inf":
        return float('inf')
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def generate_range_analysis_prompt(data: Dict[str, Any]) -> str:
    """Generate prompt"""
    frame_start = data['frame_range'][0]
    frame_end = data['frame_range'][1]
    comparisons = data['pairwise_comparisons']
    
    prompt = f"""You are a professional video data analyst. Based on analysis of consecutive frames in the range {frame_start} to {frame_end}, provide a comprehensive technical report:

## Video Segment Overview
- Frame range analyzed: {frame_start} to {frame_end}
- Total pairwise comparisons: {len(comparisons)}
- Key behavioral trends observed across frames:
"""

    gaze_angle_diffs = []
    headpose_yaw_diffs = []
    headpose_pitch_diffs = []
    headpose_roll_diffs = []
    identity_dist_diffs = []
    gaze_flag_changes = 0
    identity_match_changes = 0
    
    for comp in comparisons:
        results = comp['results']
        
        if 'gaze' in results:
            gaze = results['gaze']
            gaze_angle_diffs.append(safe_float(gaze['angle_diff']))
            if gaze['flag_changed']:
                gaze_flag_changes += 1
                
        if 'headpose' in results:
            headpose = results['headpose']
            headpose_yaw_diffs.append(safe_float(headpose['yaw_diff']))
            headpose_pitch_diffs.append(safe_float(headpose['pitch_diff']))
            headpose_roll_diffs.append(safe_float(headpose['roll_diff']))
            
        if 'identity' in results:
            identity = results['identity']
            identity_dist_diffs.append(safe_float(identity['distance_diff']))
            if identity['match_changed']:
                identity_match_changes += 1
                
    def safe_agg(values, func):
        valid = [v for v in values if not math.isinf(v)]
        return func(valid) if valid else 0.0
    
    prompt += f"""
## Aggregate Metrics
### Gaze Analysis:
- Average gaze angle difference: {safe_agg(gaze_angle_diffs, lambda x: sum(x)/len(x)):.2f}°
- Maximum gaze angle change: {safe_agg(gaze_angle_diffs, max):.2f}°
- Gaze direction changes: {gaze_flag_changes} times

### Head Pose Analysis:
- Yaw: avg {safe_agg(headpose_yaw_diffs, lambda x: sum(x)/len(x)):.2f}°, max {safe_agg(headpose_yaw_diffs, max):.2f}°
- Pitch: avg {safe_agg(headpose_pitch_diffs, lambda x: sum(x)/len(x)):.2f}°, max {safe_agg(headpose_pitch_diffs, max):.2f}°
- Roll: avg {safe_agg(headpose_roll_diffs, lambda x: sum(x)/len(x)):.2f}°, max {safe_agg(headpose_roll_diffs, max):.2f}°

### Identity Analysis:
- Average face distance difference: {safe_agg(identity_dist_diffs, lambda x: sum(x)/len(x)):.4f}
- Identity match changes: {identity_match_changes} times
"""

    prompt += "\n## Detailed Frame-to-Frame Analysis\n"
    for i, comp in enumerate(comparisons):
        frame1 = comp['frame1']
        frame2 = comp['frame2']
        results = comp['results']
        
        prompt += f"\n### Frames {frame1} → {frame2}\n"
        
        if 'gaze' in results:
            gaze = results['gaze']
            prompt += f"- Gaze: angle Δ={gaze['angle_diff']:.2f}°, "
            prompt += "direction changed" if gaze['flag_changed'] else "direction stable"
            
        if 'headpose' in results:
            headpose = results['headpose']
            prompt += f"\n- Head: yaw Δ={headpose['yaw_diff']:.2f}°, "
            prompt += f"pitch Δ={headpose['pitch_diff']:.2f}°, "
            prompt += f"roll Δ={headpose['roll_diff']:.2f}°"
            
        if 'identity' in results:
            identity = results['identity']
            prompt += f"\n- Identity: distance Δ={identity['distance_diff']:.4f}, "
            prompt += "match changed" if identity['match_changed'] else "match stable"
            
        if 'phone' in results:
            phone = results['phone']
            if phone['count_diff'] > 0:
                prompt += f"\n- Phone: +{phone['count_diff']} detected"
            elif phone['count_diff'] < 0:
                prompt += f"\n- Phone: {abs(phone['count_diff'])} disappeared"
            else:
                prompt += "\n- Phone: no change"
                
        prompt += "\n"

    prompt += """
## Technical Conclusions
- Overall behavioral patterns:
- Significant attention points:
- Recommendations for further investigation:
"""

    return prompt

def get_ai_analysis(prompt: str) -> str:
    """Send prompt to DeepSeek R1 via OpenRouter API using OpenAI client"""
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=2000
    )
    return completion.choices[0].message.content

def save_analysis_results(output_path: str, analysis: str) -> None:
    """Save analysis results to file"""
    with open(output_path, 'w') as f:
        json.dump({"analysis": analysis}, f, indent=2)

def main():
    try:
        data = load_comparison_data(r"cv_inference_project\out\all_results.json")

        prompt = generate_range_analysis_prompt(data)
        print("Generated prompt:\n", prompt)

        prompt_filename = save_prompt_to_json(prompt)
        print(f"Prompt saved to {prompt_filename}")

        print("\nGetting AI analysis...")
        analysis = get_ai_analysis(prompt)
        
        save_analysis_results("analysis_results.json", analysis)
        print("\nAnalysis saved to analysis_results.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()