from typing import Tuple
import catppuccin
from catppuccin.extras.matplotlib import load_color

def get_color(palette_name: str, color_name: str) -> str:
    identifier = getattr(catppuccin.PALETTE, palette_name).identifier
    return load_color(identifier, color_name)

def get_color_by_index(palette_name: str, index: int) -> str:
    color = list(getattr(catppuccin.PALETTE, palette_name).colors)[index]
    return get_color(palette_name, color.name.lower())
