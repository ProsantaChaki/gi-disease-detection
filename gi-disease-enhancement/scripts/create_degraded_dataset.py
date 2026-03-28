"""
Create synthetically degraded versions of the dataset
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

def add_gaussian_noise(image, sigma):
    """Add Gaussian noise"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy

def add_gaussian_blur(image, kernel_size):
    """Add Gaussian blur"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def reduce_contrast(image, gamma):
    """Reduce contrast using gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def jpeg_compression(image, quality):
    """Apply JPEG compression"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def create_degraded_versions(input_dir, output_dir):
    """
    Create multiple degradation levels
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    degradation_configs = {
        'noise_low': {'type': 'noise', 'sigma': 10},
        'noise_medium': {'type': 'noise', 'sigma': 20},
        'noise_high': {'type': 'noise', 'sigma': 30},
        'blur_low': {'type': 'blur', 'kernel': 3},
        'blur_medium': {'type': 'blur', 'kernel': 5},
        'blur_high': {'type': 'blur', 'kernel': 7},
        'contrast_low': {'type': 'contrast', 'gamma': 0.5},
        'contrast_medium': {'type': 'contrast', 'gamma': 0.7},
        'jpeg_low': {'type': 'jpeg', 'quality': 30},
        'jpeg_medium': {'type': 'jpeg', 'quality': 50},
        'combined_severe': {'type': 'combined'}  # Blur + Noise + Contrast
    }

    for deg_name, deg_config in degradation_configs.items():
        print(f"\nCreating {deg_name} degradation...")

        for split in ['train', 'val', 'test']:
            split_input = input_dir / split
            split_output = output_dir / deg_name / split

            if not split_input.exists():
                continue

            # Process each class
            for class_dir in split_input.iterdir():
                if not class_dir.is_dir():
                    continue

                output_class_dir = split_output / class_dir.name
                output_class_dir.mkdir(parents=True, exist_ok=True)

                # Process each image
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

                for img_path in tqdm(images, desc=f"{split}/{class_dir.name}"):
                    img = cv2.imread(str(img_path))

                    # Apply degradation
                    if deg_config['type'] == 'noise':
                        degraded = add_gaussian_noise(img, deg_config['sigma'])
                    elif deg_config['type'] == 'blur':
                        degraded = add_gaussian_blur(img, deg_config['kernel'])
                    elif deg_config['type'] == 'contrast':
                        degraded = reduce_contrast(img, deg_config['gamma'])
                    elif deg_config['type'] == 'jpeg':
                        degraded = jpeg_compression(img, deg_config['quality'])
                    elif deg_config['type'] == 'combined':
                        degraded = add_gaussian_blur(img, 5)
                        degraded = add_gaussian_noise(degraded, 20)
                        degraded = reduce_contrast(degraded, 0.7)

                    # Save
                    output_path = output_class_dir / img_path.name
                    cv2.imwrite(str(output_path), degraded)

    # Save configuration
    with open(output_dir / 'degradation_config.yaml', 'w') as f:
        yaml.dump(degradation_configs, f)

    print("\nDegradation complete!")

if __name__ == '__main__':
    input_dir = Path('/workspace/data/splits')
    output_dir = Path('/workspace/data/degraded')

    create_degraded_versions(input_dir, output_dir)