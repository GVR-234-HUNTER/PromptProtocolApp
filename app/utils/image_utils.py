import base64
import os
import io
from PIL import Image
from typing import List, Optional, Tuple


def image_to_base64(image_path: str) -> Optional[str]:
    """
    Convert an image file to base64 encoded string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string representation of the image or None if failed
    """
    try:
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist")
            return None

        # Open the image file
        with open(image_path, 'rb') as img_file:
            # Read the file content and encode it to base64
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode('utf-8')
            return base64_str
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None


def images_to_base64(image_paths: List[str]) -> List[str]:
    """
    Convert multiple image files to base64 encoded strings.

    Args:
        image_paths: List of paths to image files

    Returns:
        List of base64 encoded strings
    """
    base64_images = []
    for path in image_paths:
        base64_str = image_to_base64(path)
        if base64_str:
            base64_images.append(base64_str)
    return base64_images


def resize_image_if_needed(img: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Resize an image if it exceeds the maximum dimensions.

    Args:
        img: PIL Image object
        max_size: Maximum (width, height) tuple

    Returns:
        Resized PIL Image object
    """
    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def optimize_image_for_api(image_path: str) -> Optional[str]:
    """
    Optimize image for API usage by resizing if needed and converting to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Optimized base64 encoded string
    """
    try:
        # Open and resize the image if needed
        img = Image.open(image_path)
        img = resize_image_if_needed(img)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to bytes IO and convert to base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_bytes = img_byte_arr.getvalue()

        # Encode as base64
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"Error optimizing image: {str(e)}")
        return None
