import pytesseract
import cv2
from PIL import ImageFont, ImageDraw, Image
import re
import datetime
from typing import Callable
import numpy as np
import sys
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# img_path = "image2.png"
if len(sys.argv) < 2:
    print("Usage: python newslater.py <image_path>")
    sys.exit(1)
else:
    img_path = sys.argv[1]

img_cv = cv2.imread(img_path)
img_height, img_width, *_ = img_cv.shape

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
boxes: dict = pytesseract.image_to_boxes(img_rgb, output_type=pytesseract.Output.DICT)
data: dict = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)

img = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(img)
font_src = ImageFont.truetype("comicsans.ttf", 100)

def get_new_date():
    today = datetime.date.today()
    today += datetime.timedelta(days=3)
    return str(today)


def replace_ed(match: re.Match[str]) -> str:
    return match.group()[:-2]

rules_data: dict[str, str | Callable[[re.Match[str]], str]] = {
    r"^(was|were)$": "will be",
    r"^after$": "before",
    r"^have$": "will",
    r"[0-9]+[/\-.][0-9]+[/\-.][0-9]+": get_new_date(),
    r".*ed$": replace_ed
}

rules = {
    re.compile(rule): subs
    for rule, subs in rules_data.items()
}


def get_font_size(draw: ImageDraw.ImageDraw, width: int, height: int, text: str, font_src: ImageFont.FreeTypeFont) -> int:
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

for (
    level,
    page_num, block_num,
    par_num, line_num, word_num,
    left, top, width, height,
    conf, text
) in tqdm(zip(*data.values())):
    if conf < 0:
        continue
    
    for rule, subs in rules.items():

        new_text, n = rule.subn(subs, text.lower())
        if n > 0:
            font_size = get_font_size(draw, width, height, new_text, font_src)
            font = font_src.font_variant(size=font_size)

            draw.rectangle([(left, top), (left+width, top+height)], fill="#FFF")
            pos = (left+width*0.5, top+height*0.5)
            draw.text(pos, new_text, fill="#000", font=font, anchor="mm")
            break

img.save("later.png")
img.show()
