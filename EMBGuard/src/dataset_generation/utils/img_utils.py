from PIL import Image
import glob
from tqdm import tqdm
import base64
import io
import os
import mimetypes
import json

def crop_image_center(image_path: str):
    image = Image.open(image_path)

    width, height = image.size

    left = width // 3 
    right = 2 * (width // 3) 
    top = 0  
    bottom = height  

    center_image = image.crop((left, top, right, bottom))

    center_image_path = "/data/hssd-hab/objects_cropped_img/" + image_path.split("/")[-1]
    center_image.save(center_image_path)


def encode_image_to_base64(image_path: str) -> str:
    if not image_path:
        return None
    
    # Convert to absolute path if relative
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        image_data = f.read()
    return f"data:{mime_type};base64," + base64.b64encode(image_data).decode("utf-8")


def compress_and_encode_image_to_base64(image_path: str, max_size=(512, 512), quality=85) -> str:
    """
    - max_size: (width, height) (default: 512x512)
    - quality: JPEG compression quality (0~100, default: 85)
    """
    if not image_path:
        return None
    
    # Convert to absolute path if relative
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")  
        img.thumbnail(max_size)   
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        encoded_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return "data:image/jpeg;base64," + encoded_str


def add_image_paths_to_scenarios(
    text_file: str,
    output_dir: str,
    output_with_images: str,
    merge_source: str
):
    """
    Add image paths to scenarios in a text file.
    
    Args:
        text_file: Path to input text file containing scenarios
        output_dir: Directory containing downloaded images
        output_with_images: Path to output file for texts_with_images_*.json
        merge_source: Merge source name
    """
    # Load scenarios
    with open(text_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenarios = data.get('scenarios', data) if isinstance(data, dict) else data
    
    # Add image paths
    for scenario in scenarios:
        scenario_id = scenario.get('id', '')
        merge_source_from_scenario = scenario.get('merge_source', '')
        
        # Look for image files matching the pattern
        pattern = f'{output_dir}/{scenario_id}-{merge_source_from_scenario}.*'
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # Use the first matching file
            image_path = matching_files[0]
            scenario['image_path'] = image_path
            scenario['image_saved'] = True
            scenario['image_format'] = os.path.splitext(image_path)[1][1:]  # Remove dot
        else:
            scenario['image_path'] = None
            scenario['image_saved'] = False
    
    # Save updated scenarios
    output_data = {
        'metadata': {
            'total_scenarios': len(scenarios),
            'scenarios_with_images': sum(1 for s in scenarios if s.get('image_saved', False)),
            'merge_source': merge_source
        },
        'scenarios': scenarios
    }
    
    with open(output_with_images, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'    Added image paths to {len(scenarios)} scenarios')
    print(f'    Images found: {output_data["metadata"]["scenarios_with_images"]}')