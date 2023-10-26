import base64
from pathlib import Path

import copy
import imageio.v3 as imageio
import numpy as np
from PIL import Image, PngImagePlugin
from modules import images, shared
from modules.processing import Processed, StableDiffusionProcessing

# from scripts.animatediff_logger import logger_animatediff as logger
# from scripts.animatediff_ui import AnimateDiffProcess

try:
    from scripts.animatediff_logger import logger_animatediff as logger
    from scripts.animatediff_ui import AnimateDiffProcess
except ImportError:
    from scripts.animatediff.animatediff_logger import logger_animatediff as logger
    from scripts.animatediff.animatediff_ui import AnimateDiffProcess


class AnimateDiffOutput:
    def output(
        self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess, select = False
    ):
        video_paths = []
        logger.info("Merging images into GIF.")
        Path(f"{p.outpath_samples}/AnimateDiff").mkdir(exist_ok=True, parents=True)
        step = params.video_length if params.video_length > params.batch_size else params.batch_size
        
        origin_video = []
        index = 0
        for i in range(res.index_of_first_image, len(res.images), step):
            # frame interpolation replaces video_list with interpolated frames
            # so make a copy instead of a slice (reference), to avoid modifying res
            video_list = [image.copy() for image in res.images[i : i + params.video_length]]

            seq = images.get_next_sequence_number(f"{p.outpath_samples}/AnimateDiff", "")
            filename = f"{seq:05}-{res.seed}"
            video_path_prefix = f"{p.outpath_samples}/AnimateDiff/{filename}"

            video_list = self._add_reverse(params, video_list)
            video_list = self._interp(p, params, video_list, filename)
            video_paths += self._save(params, video_list, video_path_prefix, res, i)
            if index == 0:
                origin_video = copy.deepcopy(video_list)
            index += 1

        if len(video_paths) > 0:
            # if not p.is_api:
            #     res.images = video_paths
            # else:
                # res.images = self._encode_video_to_b64(video_paths)
            res.images = origin_video

    def _add_reverse(self, params: AnimateDiffProcess, video_list: list):
        if 0 in params.reverse:
            video_list_reverse = video_list[::-1]
            if 1 in params.reverse:
                video_list_reverse.pop(0)
            if 2 in params.reverse:
                video_list_reverse.pop(-1)
            return video_list + video_list_reverse
        return video_list

    def _interp(
        self,
        p: StableDiffusionProcessing,
        params: AnimateDiffProcess,
        video_list: list,
        filename: str
    ):
        if params.interp not in ['FILM']:
            return video_list
        
        try:
            from deforum_helpers.frame_interpolation import (
                calculate_frames_to_add, check_and_download_film_model)
            from film_interpolation.film_inference import run_film_interp_infer
        except ImportError:
            logger.error("Deforum not found. Please install: https://github.com/deforum-art/deforum-for-automatic1111-webui.git")
            return video_list

        import glob
        import os
        import shutil

        import modules.paths as ph

        # load film model
        deforum_models_path = ph.models_path + '/Deforum'
        film_model_folder = os.path.join(deforum_models_path,'film_interpolation')
        film_model_name = 'film_net_fp16.pt'
        film_model_path = os.path.join(film_model_folder, film_model_name)
        check_and_download_film_model('film_net_fp16.pt', film_model_folder)

        film_in_between_frames_count = calculate_frames_to_add(len(video_list), params.interp_x) 

        # save original frames to tmp folder for deforum input
        tmp_folder = f"{p.outpath_samples}/AnimateDiff/tmp"
        input_folder = f"{tmp_folder}/input"
        os.makedirs(input_folder, exist_ok=True)
        for tmp_seq, frame in enumerate(video_list):
            imageio.imwrite(f"{input_folder}/{tmp_seq:05}.png", frame)

        # deforum saves output frames to tmp/{filename}
        save_folder = f"{tmp_folder}/{filename}"
        os.makedirs(save_folder, exist_ok=True)

        run_film_interp_infer(
            model_path = film_model_path,
            input_folder = input_folder,
            save_folder = save_folder,
            inter_frames = film_in_between_frames_count)

        # load deforum output frames and replace video_list
        interp_frame_paths = sorted(glob.glob(os.path.join(save_folder, '*.png')))
        video_list = []
        for f in interp_frame_paths:
            with Image.open(f) as img:
                img.load()
                video_list.append(img)
        
        # if saving PNG, also save interpolated frames
        if "PNG" in params.format:
            save_interp_path = f"{p.outpath_samples}/AnimateDiff/interp"
            os.makedirs(save_interp_path, exist_ok=True)
            shutil.move(save_folder, save_interp_path)

        # remove tmp folder
        try: shutil.rmtree(tmp_folder)
        except OSError as e: print(f"Error: {e}")

        return video_list

    def _save(
        self,
        params: AnimateDiffProcess,
        video_list: list,
        video_path_prefix: str,
        res: Processed,
        index: int,
    ):
        video_paths = []
        video_array = [np.array(v) for v in video_list]
        infotext = res.info
        use_infotext = shared.opts.enable_pnginfo and infotext is not None
        if "PNG" in params.format and shared.opts.data.get("animatediff_save_to_custom", False):
            Path(video_path_prefix).mkdir(exist_ok=True, parents=True)
            for i, frame in enumerate(video_list):
                png_filename = f"{video_path_prefix}/{i:05}.png"
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text('parameters', res.infotexts[0])
                imageio.imwrite(png_filename, frame, pnginfo=png_info)

        if "GIF" in params.format:
            video_path_gif = video_path_prefix + ".gif"
            video_paths.append(video_path_gif)
            if shared.opts.data.get("animatediff_optimize_gif_palette", False):
                try:
                    import av
                except ImportError:
                    from launch import run_pip
                    run_pip(
                        "install imageio[pyav]",
                        "sd-webui-animatediff GIF palette optimization requirement: imageio[pyav]",
                    )
                imageio.imwrite(
                    video_path_gif, video_array, plugin='pyav', fps=params.fps, 
                    codec='gif', out_pixel_format='pal8',
                    filter_graph=(
                        {
                            "split": ("split", ""),
                            "palgen": ("palettegen", ""),
                            "paluse": ("paletteuse", ""),
                            "scale": ("scale", f"{video_list[0].width}:{video_list[0].height}")
                        },
                        [
                            ("video_in", "scale", 0, 0),
                            ("scale", "split", 0, 0),
                            ("split", "palgen", 1, 0),
                            ("split", "paluse", 0, 0),
                            ("palgen", "paluse", 0, 1),
                            ("paluse", "video_out", 0, 0),
                        ]
                    )
                )
                # imageio[pyav].imwrite doesn't support comment parameter
                if use_infotext:
                    try:
                        import exiftool
                    except ImportError:
                        from launch import run_pip
                        run_pip(
                            "install PyExifTool",
                            "sd-webui-animatediff GIF palette optimization requirement: PyExifTool",
                        )
                        import exiftool
                    finally:
                        try:
                            exif_tool = exiftool.ExifTool()
                            with exif_tool:
                                escaped_infotext = infotext.replace('\n', r'\n')
                                exif_tool.execute(f"-Comment={escaped_infotext}", video_path_gif)
                        except FileNotFoundError:
                            logger.warn(
                                "exiftool not found, required for infotext with optimized GIF palette, try: apt install libimage-exiftool-perl or https://exiftool.org/"
                            )
            else:
                imageio.imwrite(
                    video_path_gif,
                    video_array,
                    plugin='pillow',
                    duration=(1000 / params.fps),
                    loop=params.loop_number,
                    comment=(infotext if use_infotext else "")
                )
            if shared.opts.data.get("animatediff_optimize_gif_gifsicle", False):
                self._optimize_gif(video_path_gif)
        if "MP4" in params.format:
            video_path_mp4 = video_path_prefix + ".mp4"
            video_paths.append(video_path_mp4)
            try:
                imageio.imwrite(video_path_mp4, video_array, fps=params.fps, codec="h264")
            except:
                from launch import run_pip
                run_pip(
                    "install imageio[ffmpeg]",
                    "sd-webui-animatediff save mp4 requirement: imageio[ffmpeg]",
                )
                imageio.imwrite(video_path_mp4, video_array, fps=params.fps, codec="h264")
        if "TXT" in params.format and res.images[index].info is not None:
            video_path_txt = video_path_prefix + ".txt"
            self._save_txt(video_path_txt, infotext)
        return video_paths

    def _optimize_gif(self, video_path: str):
        try:
            import pygifsicle
        except ImportError:
            from launch import run_pip

            run_pip(
                "install pygifsicle",
                "sd-webui-animatediff GIF optimization requirement: pygifsicle",
            )
            import pygifsicle
        finally:
            try:
                pygifsicle.optimize(video_path)
            except FileNotFoundError:
                logger.warn("gifsicle not found, required for optimized GIFs, try: apt install gifsicle")

    def _save_txt(
        self,
        video_path: str,
        info: str,
    ):
        with open(video_path, "w", encoding="utf8") as file:
            file.write(f"{info}\n")

    def _encode_video_to_b64(self, paths):
        videos = []
        for v_path in paths:
            with open(v_path, "rb") as video_file:
                encoded_video = base64.b64encode(video_file.read())
            videos.append(encoded_video.decode("utf-8"))
        return videos
