import os

from modules.paths import data_path
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img

# from scripts.animatediff_ui import AnimateDiffProcess

try:
    from scripts.animatediff_ui import AnimateDiffProcess
except ImportError:
    from scripts.animatediff.animatediff_ui import AnimateDiffProcess


def update_infotext(p: StableDiffusionProcessing, params: AnimateDiffProcess):
    if p.extra_generation_params is not None:
        p.extra_generation_params["AnimateDiff"] = params.get_dict(isinstance(p, StableDiffusionProcessingImg2Img))


def write_params_txt(info: str):
    with open(os.path.join(data_path, "params.txt"), "w", encoding="utf8") as file:
        file.write(info)
