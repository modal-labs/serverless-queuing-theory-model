import os
from PIL import Image

for fn in os.listdir("."):
    stem, suffix = os.path.splitext(fn)
    if suffix != ".png":
        continue

    img = Image.open(fn).convert("RGB")
    img.save(stem + ".jpg", quality=20)

