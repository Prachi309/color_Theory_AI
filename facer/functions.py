import argparse
import glob
import os
import os.path as osp
import random
from collections import Counter
import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv() 

api_key = os.getenv("API_KEY")

def compress_image(image_path, max_size=800, quality=85):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        compressed_path = image_path.replace('.jpg', '_compressed.jpg')
        cv2.imwrite(compressed_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return compressed_path
    except Exception as e:
        print(f"Compression error: {e}")
        return image_path

def cleanup_temp_files(*file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Cleanup error for {file_path}: {e}")

def _lazy_import_torch():
    if not hasattr(_lazy_import_torch, '_torch'):
        print("Loading torch (lazy loading)...")
        import torch
        _lazy_import_torch._torch = torch
    return _lazy_import_torch._torch

def _lazy_import_facer():
    if not hasattr(_lazy_import_facer, '_facer'):
        print("Loading facer (lazy loading)...")
        import facer
        _lazy_import_facer._facer = facer
    return _lazy_import_facer._facer

def _lazy_import_mediapipe():
    if not hasattr(_lazy_import_mediapipe, '_mp'):
        print("Loading mediapipe (lazy loading)...")
        import mediapipe as mp
        _lazy_import_mediapipe._mp = mp
    return _lazy_import_mediapipe._mp

def _lazy_import_torchvision():
    if not hasattr(_lazy_import_torchvision, '_torchvision'):
        print("Loading torchvision (lazy loading)...")
        import torchvision.transforms as transforms
        _lazy_import_torchvision._torchvision = transforms
    return _lazy_import_torchvision._torchvision

def _lazy_import_pil():
    if not hasattr(_lazy_import_pil, '_pil'):
        print("Loading PIL (lazy loading)...")
        from PIL import Image
        _lazy_import_pil._pil = Image
    return _lazy_import_pil._pil

def _lazy_import_skimage():
    if not hasattr(_lazy_import_skimage, '_skimage'):
        print("Loading skimage (lazy loading)...")
        from skimage.filters import gaussian
        _lazy_import_skimage._skimage = gaussian
    return _lazy_import_skimage._skimage

def _lazy_import_scipy():
    if not hasattr(_lazy_import_scipy, '_scipy'):
        print("Loading scipy (lazy loading)...")
        from scipy import stats
        _lazy_import_scipy._scipy = stats
    return _lazy_import_scipy._scipy

def _lazy_import_cv2():
    if not hasattr(_lazy_import_cv2, '_cv2'):
        print("Loading cv2 (lazy loading)...")
        import cv2
        _lazy_import_cv2._cv2 = cv2
    return _lazy_import_cv2._cv2

def _lazy_import_np():
    if not hasattr(_lazy_import_np, '_np'):
        print("Loading numpy (lazy loading)...")
        import numpy as np
        _lazy_import_np._np = np
    return _lazy_import_np._np


def get_rgb_codes(path):
    compressed_path = compress_image(path, max_size=600, quality=80)
    
    torch = _lazy_import_torch()
    cv2 = _lazy_import_cv2()
    np = _lazy_import_np()
    import gc
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from lightweight_face_parser import LightweightFaceParser
    
    try:
        sample = cv2.imread(compressed_path)
        img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device=device)
       
        face_parser = LightweightFaceParser(device=device)
        
        data = {'image_ids': torch.tensor([0])}
        
        with torch.inference_mode():
            result = face_parser.forward(image_tensor, data)
        
        seg_logits = result['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  
        seg_probs = seg_probs.cpu()
        
        lip_prob = seg_probs[0, 1]  
        binary_mask = (lip_prob >= 0.3).numpy().astype(int) 
        
        indices = np.argwhere(binary_mask)
        if len(indices) > 0:
            rgb_codes = img[indices[:, 0], indices[:, 1], :]
        else:
            h, w = img.shape[:2]
            lower_region = img[int(h*0.6):, :, :]
            rgb_codes = lower_region.reshape(-1, 3)
        
        del image_tensor, seg_logits, seg_probs, lip_prob, binary_mask, indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
      
        gc.collect()
        
        return rgb_codes
        
    except Exception as e:
        print(f"Error in get_rgb_codes: {e}")
        return np.array([])
    finally:
        cleanup_temp_files(compressed_path)

def filter_lip_random(rgb_codes,randomNum=40):
    blue_condition = (rgb_codes[:, 2] <= 227)
    red_condition = (rgb_codes[:, 0] >= 97)
    filtered_rgb_codes = rgb_codes[blue_condition & red_condition]
    step = max(1, filtered_rgb_codes.shape[0] // randomNum)
    indices = np.arange(0, filtered_rgb_codes.shape[0], step)[:randomNum]
    random_rgb_codes = filtered_rgb_codes[indices]
    return random_rgb_codes


def calc_dis(rgb_codes):
    spring = [[253,183,169],[247,98,77],[186,33,33]]
    summer = [[243,184,202],[211,118,155],[147,70,105]]
    autum = [[210,124,110],[155,70,60],[97,16,28]]
    winter = [[237,223,227],[177,47,57],[98,14,37]]
  
    res = []
    for i in range(len(rgb_codes)):
      sp = np.inf
      su = np.inf
      au = np.inf
      win = np.inf
      for j in range(3):
        sp = min(sp, np.linalg.norm(rgb_codes[i] - spring[j]))
        su = min(su, np.linalg.norm(rgb_codes[i]- summer[j]))
        au = min(au, np.linalg.norm(rgb_codes[i] - autum[j]))
        win = min(win, np.linalg.norm(rgb_codes[i] - winter[j]))
    
      min_type = min(sp, su, au, win)
      if min_type == sp:
        ctype = "sp"
      elif min_type == su:
        ctype = "su"
      elif min_type == au:
        ctype = "au"
      elif min_type == win:
        ctype = "win"
    
      res.append(ctype)
    return res


def save_skin_mask(img_path):
    torch = _lazy_import_torch()
    facer = _lazy_import_facer()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)  
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
      faces = face_detector(image)

    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
      faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu() 
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    face_skin = tensor[:, :, 1]
    binary_mask = (face_skin >= 0.5).astype(int)

    sample = cv2.imread(img_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    masked_image = np.zeros_like(img) 
    try: 
      masked_image[binary_mask == 1] = img[binary_mask == 1] 
      masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB)
      cv2.imwrite("temp.jpg" , masked_image)
    except:
      print("error occurred")


def get_eye_color(image_path):
    mp = _lazy_import_mediapipe()
    cv2 = _lazy_import_cv2()
    np = _lazy_import_np()
    stats = _lazy_import_scipy()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    LEFT_IRIS = [468, 469, 470, 471]

    def isolate_region(image, landmarks, indices):
        h, w = image.shape[:2]
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        region = cv2.bitwise_and(image, image, mask=mask)
        return region, mask

    def classify_eye_color(hue, saturation, value, rgb=None):
        if rgb is not None:
            r, g, b = rgb
            if max(r, g, b) < 90 and abs(r - g) < 15 and abs(g - b) < 15 and abs(b - r) < 15:
                return "Black"
        if value < 40:
            return "Black"
        if (hue <= 15 or hue >= 165) and value < 65:
            return "Black"
        if value > 200 and saturation < 30:
            return "Gray"
        if 10 < hue <= 25 and saturation > 50:
            return "Amber"
        if 25 < hue <= 45:
            return "Hazel"
        if 45 < hue <= 85:
            return "Green"
        if 85 < hue <= 130:
            return "Blue"
        if hue <= 10 or hue >= 160:
            return "Brown"
        if 140 <= hue <= 160 and saturation < 50:
            return "Violet"
        return "Unknown"

    def get_filtered_dominant_color(image, mask):
        pixels = image[mask == 255]
        if len(pixels) == 0:
            return "Unknown", (0, 0, 0)
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        filtered = hsv[(hsv[:, 1] > 30) & (hsv[:, 2] > 50) & (hsv[:, 2] < 230)]
        if len(filtered) == 0:
            return "Unknown", (0, 0, 0)
        hue_mode = int(stats.mode(filtered[:, 0], keepdims=False)[0])
        sat_avg = int(np.mean(filtered[:, 1]))
        val_avg = int(np.mean(filtered[:, 2]))
        rgb = cv2.cvtColor(np.uint8([[[hue_mode, sat_avg, val_avg]]]), cv2.COLOR_HSV2BGR)[0][0]
        rgb_tuple = tuple(int(c) for c in rgb)
        color_name = classify_eye_color(hue_mode, sat_avg, val_avg, rgb=rgb_tuple)
        return color_name, rgb_tuple

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        _, left_iris_mask = isolate_region(image, face_landmarks.landmark, LEFT_IRIS)
        color_name, rgb_tuple = get_filtered_dominant_color(image, left_iris_mask)
        return {"rgb": rgb_tuple, "color": color_name}
    else:
        return None


def analyze_lip_color(image_path):
    rgb_codes = get_rgb_codes(image_path)
    if rgb_codes is None or len(rgb_codes) == 0:
        return {"error": "No lip region detected"}
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for k-means clustering. Please add it to requirements.txt.")
    kmeans = KMeans(n_clusters=1, random_state=42).fit(rgb_codes)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    dominant_color_rgb = tuple(int(x) for x in dominant_color)
    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb

    filtered = filter_lip_random(rgb_codes, 40)
    types = Counter(calc_dis(filtered))
    season = max(types, key=types.get)

    return {
        "dominant_color_rgb": dominant_color_rgb,
        "dominant_color_hex": dominant_color_hex,
        "season": season
    }


def analyze_skin_color(image_path):
    torch = _lazy_import_torch()
    facer = _lazy_import_facer()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu()
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()
    face_skin = tensor[:, :, 1]
    binary_mask = (face_skin >= 0.5).astype(int)
    sample = cv2.imread(image_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    skin_pixels = img[binary_mask == 1]
    if skin_pixels is None or len(skin_pixels) == 0:
        return {"error": "No skin region detected"}
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for k-means clustering. Please add it to requirements.txt.")
    kmeans = KMeans(n_clusters=1, random_state=42).fit(skin_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    dominant_color_rgb = tuple(int(x) for x in dominant_color)
    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb
    return {
        "dominant_color_rgb": dominant_color_rgb,
        "dominant_color_hex": dominant_color_hex
    }


def analyze_hair_color(image_path):
    torch = _lazy_import_torch()
    facer = _lazy_import_facer()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu()
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()
    print('Segmentation tensor shape:', tensor.shape)
    num_classes = tensor.shape[2] if tensor.ndim == 3 else 0
    print('Number of available classes:', num_classes)
    for idx in range(num_classes):
        mask = tensor[:, :, idx]
        print(f'Class {idx} mean mask value:', mask.mean())
    hair_mask = tensor[:, :, -1]
    binary_mask = (hair_mask >= 0.5).astype(int)
    sample = cv2.imread(image_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    hair_pixels = img[binary_mask == 1]
    if hair_pixels is None or len(hair_pixels) == 0:
        return {"error": "No hair region detected"}
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for k-means clustering. Please add it to requirements.txt.")
    kmeans = KMeans(n_clusters=1, random_state=42).fit(hair_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    dominant_color_rgb = tuple(int(x) for x in dominant_color)
    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb
    return {
        "dominant_color_rgb": dominant_color_rgb,
        "dominant_color_hex": dominant_color_hex
    }

    import requests

    def get_palette_from_llm(prompt, api_key):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",  
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        print("OpenRouter response:", response.status_code, response.text)  
        return response.json()


def get_style_recommendation_from_llm(answers, api_key):
    """
    Given a dict of answers (with keys: dressing_focus, gender, body_type, context_answer),
    build a prompt, call OpenRouter, and return the suggestion string.
    """
    dressing_focus = answers.get("dressing_focus", "")
    gender = answers.get("gender", "")
    body_type = answers.get("body_type", "")
    context_answer = answers.get("context_answer", "")
    prompt = f"""
Recommend 3 items each of:
- Clothing
- Footwear
- Accessories (if suitable)
- Makeup products (foundation, lipstick, blush)
- Fragrance

For a user with:
Dressing focus: {dressing_focus}
Gender: {gender}
Body type: {body_type}
Context: {context_answer}

Return as JSON array only:
[
  {{
    "category": "...",
    "product": "...",
    "description": "...",
    "query": "search keywords only for finding the product"
  }},
  ...
]
"""
    llm_response = get_palette_from_llm(prompt, api_key)
    try:
        suggestion = llm_response['choices'][0]['message']['content']
    except Exception as e:
        suggestion = str(llm_response)
    return suggestion