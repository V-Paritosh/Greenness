import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import pillow_heif

def load_image_any_format(path):
    if path.lower().endswith(".heic"):
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(path)

def calculate_greenness_intensity(image_path, show_visuals=True):
    # Load image (BGR)
    img = load_image_any_format(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract H, S, V channels
    h, s, v = cv2.split(hsv)

    # Normalize Hue to [0,180] → [0,1]
    h = h.astype(np.float32) / 180.0
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0

    # Define green hue range (approx. 40°–75° → 40/180–75/180 in normalized hue)
    lower_hue, upper_hue = 40/180.0, 75/180.0

    # Build a 2D "greenness map"
    greenness_map = np.zeros_like(h, dtype=np.float32)

    # Pixels within green hue range → greenness proportional to saturation * value
    mask = (h >= lower_hue) & (h <= upper_hue)
    greenness_map[mask] = s[mask] * v[mask]

    # Calculate average greenness (0–1 scale)
    avg_greenness = greenness_map.mean() * 100  # convert to % scale

    if show_visuals:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Greenness Map (intensity)")
        plt.imshow(greenness_map, cmap="Greens")
        plt.colorbar(label="Greenness Intensity")
        plt.axis("off")

        plt.show()

    return avg_greenness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate plant greenness intensity from an image.")
    parser.add_argument("image_path", type=str, help="Path to the plant image file")
    parser.add_argument("--no-show", action="store_true", help="Disable visualization of results")

    args = parser.parse_args()

    score = calculate_greenness_intensity(args.image_path, show_visuals=not args.no_show)
    print(f"Average Greenness Score: {score:.2f}%")
