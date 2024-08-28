import cv2
import numpy as np

def preprocess_image(image, weather_condition):
    if weather_condition == 'fog':
        return enhance_contrast_for_fog(image)
    elif weather_condition == 'rain':
        return apply_rain_effects(image)
    elif weather_condition == 'snow':
        return apply_snow_effects(image)
    else:
        return image  # No preprocessing for clear weather

def enhance_contrast_for_fog(image):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_rain_effects(image):
    # Example: simple blur to simulate rain
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_snow_effects(image):
    # Example: adding white noise to simulate snow
    snow = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    snow_mask = (snow > 240).astype(np.uint8) * 255
    return cv2.addWeighted(image, 0.9, snow_mask, 0.1, 0)
