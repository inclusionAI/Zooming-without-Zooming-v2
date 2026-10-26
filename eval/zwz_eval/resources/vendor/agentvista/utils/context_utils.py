"""Context utilities: token estimation, image processing, base64 data URI."""

import math
import io
import base64
import re
from PIL import Image

def estimate_tokens(text: str) -> int:
    """Rough token count for context management."""
    return len(text) // 4


def rebuild_images_from_history(conversation_history, image_map):
    """Rebuild list of images in order of [Image: ...] refs in conversation."""
    out = []
    for msg in conversation_history:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
            continue
        refs = re.findall(r"\[Image: ([^\]]+)\]", msg["content"])
        for ref in refs:
            key = ref.split(", file:")[0].strip() if ", file:" in ref else ref.strip()
            if key in image_map:
                out.append(image_map[key])
    return out


def pil_to_base64_data_uri(pil_img, max_pixels=2000000, min_pixels=40000, quality=85):
    """Convert PIL image to base64 data URI for API."""
    if pil_img.mode == "RGBA":
        from PIL import Image as PILImage
        bg = PILImage.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    processed = process_image(pil_img, max_pixels=max_pixels, min_pixels=min_pixels)
    buf = io.BytesIO()
    processed.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def process_image(image: Image.Image, max_pixels: int = 2000000, min_pixels: int = 40000, use_lanczos: bool = True) -> Image.Image:
    """Resize image to stay within pixel range."""
    resample = Image.Resampling.LANCZOS if use_lanczos else Image.Resampling.NEAREST
    if image.width * image.height > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height))
        w, h = int(image.width * f), int(image.height * f)
        image = image.resize((w, h), resample=resample)
    if image.width * image.height < min_pixels:
        f = math.sqrt(min_pixels / (image.width * image.height))
        w, h = int(image.width * f), int(image.height * f)
        image = image.resize((w, h), resample=resample)
    if image.width < 28 or image.height < 28:
        f = 28 / min(image.width, image.height)
        w, h = int(image.width * f + 1), int(image.height * f + 1)
        image = image.resize((w, h), resample=resample)
    if image.width / image.height >= 200:
        image = image.resize((image.width, int(image.width / 190 + 1)), resample=resample)
    if image.height / image.width >= 200:
        image = image.resize((int(image.height / 190 + 1), image.height), resample=resample)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
