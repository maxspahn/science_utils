from typing import Tuple
import catppuccin
from catppuccin.extras.matplotlib import load_color

def get_color(palette_name: str, color_name: str) -> str:
    identifier = getattr(catppuccin.PALETTE, palette_name).identifier
    return load_color(identifier, color_name)
