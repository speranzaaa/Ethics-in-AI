import os

FONT_NAME = "Body"


def setup_fonts(pdf):
    from ..config import DEJAVU_FONTS

    base_path = DEJAVU_FONTS.get("")
    if base_path and os.path.exists(base_path):
        for style, path in DEJAVU_FONTS.items():
            if os.path.exists(path):
                pdf.add_font(FONT_NAME, style=style, fname=path)
        return FONT_NAME
    return "Helvetica"
