import os

from modules.paths import data_path
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img

from .animatediff_ui import AnimateDiffProcess
from .animatediff_logger import logger_animatediff as logger


def update_infotext(p: StableDiffusionProcessing, params: AnimateDiffProcess):
    if p.extra_generation_params is not None:
        p.extra_generation_params["AnimateDiff"] = params.get_dict(isinstance(p, StableDiffusionProcessingImg2Img))


def write_params_txt(info: str):
    with open(os.path.join(data_path, "params.txt"), "w", encoding="utf8") as file:
        file.write(info)



def infotext_pasted(infotext, results):
    for k, v in results.items():
        if not k.startswith("AnimateDiff"):
            continue

        assert isinstance(v, str), f"Expect string but got {v}."
        try:
            for items in v.split(', '):
                field, value = items.split(': ')
                results[f"AnimateDiff {field}"] = value
        except Exception:
            logger.warn(
                f"Failed to parse infotext, legacy format infotext is no longer supported:\n{v}"
            )
        break
