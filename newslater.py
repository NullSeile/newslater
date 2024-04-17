import pytesseract
import cv2
from PIL import ImageFont, ImageDraw, Image
import re
import datetime
from typing import Callable
import numpy as np
import sys
import pandas as pd
# from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

if len(sys.argv) < 2:
    print("Usage: python newslater.py <image_path>")
    sys.exit(1)

img_cv = cv2.imread(sys.argv[1])
img_height, img_width, *_ = img_cv.shape

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
boxes: dict = pytesseract.image_to_boxes(img_rgb, output_type=pytesseract.Output.DICT)
data: pd.DataFrame = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DATAFRAME)

data = data[data["text"].notna()]
data = data.reset_index()

data["len"] = data["text"].apply(len)

# Increase the length by 1 to include the space
data["len"] += 1
data.loc[0, "len"] -= 1 # type: ignore

data["char_end"] = data["len"].cumsum()
data["char_start"] = data["char_end"].shift(1).fillna(0).astype(np.int64)
data["char_start"] += 1
data.loc[0, "char_start"] -= 1 # type: ignore

data = data.drop(columns=["len"])

full_text = " ".join(data["text"])


img = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(img)
font_src = ImageFont.truetype("comicsans.ttf", 100)

def get_new_date():
    today = datetime.date.today()
    today += datetime.timedelta(days=3)
    return str(today)


def replace_ed(match: re.Match[str]) -> str:
    return f"{match.group()[:-2]}"
    # return f"will {match.group()[:-2]}"


rules: dict[str, str | Callable[[re.Match[str]], str]] = {
    r"\b(was|were)\b": "will be",
    r"\bafter\b": "before",
    r"\bhave\b": "will",
    r"[0-9]+[/\-.][0-9]+[/\-.][0-9]+": get_new_date(),
    r"\w*ed\b": replace_ed
}


def get_font_size(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    text: str,
    font_src: ImageFont.FreeTypeFont
) -> int:
    def get_length(size: float) -> float:
        font = font_src.font_variant(size=size)
        return draw.textlength(text, font=font)

    size = 20
    sign = np.sign(width - get_length(size))
    if sign == 0:
        return size
    
    while (
        sign == np.sign(width - get_length(size)) and 
        (size < height or sign == -1)
    ):
        size += sign

    return size

def get_bounds(data: pd.DataFrame) -> tuple[int, int, int, int]:
    left = data["left"].min()
    top = data["top"].min()
    right = (data["left"] + data["width"]).max()
    bottom = (data["top"] + data["height"]).max()
    return left, top, right, bottom


for rule, subs in rules.items():
    for m in re.finditer(rule, full_text, flags=re.IGNORECASE):
        words = data[
            (data["char_start"] >= m.start()) & 
            (data["char_end"] <= m.end())
        ]

        left, top, right, bottom = get_bounds(words)

        new_text, n = re.subn(rule, subs, full_text[m.start():m.end()], flags=re.IGNORECASE)
        font_size = get_font_size(draw, right-left, top-bottom, new_text, font_src)
        font = font_src.font_variant(size=font_size)

        draw.rectangle([left, top, right, bottom], fill="#FFF")
        pos = ((left+right)/2, (top+bottom)/2)
        draw.text(pos, new_text, fill="#000", font=font, anchor="mm")

img.save("later.png")
img.show()
