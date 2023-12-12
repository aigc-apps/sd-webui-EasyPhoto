import base64
import datetime
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from PIL import Image, PngImagePlugin
import PIL.features
import piexif
from modules import images, shared
from modules.processing import Processed, StableDiffusionProcessing

from .animatediff_logger import logger_animatediff as logger
from .animatediff_ui import AnimateDiffProcess



class AnimateDiffOutput:
    api_encode_pil_to_base64_hooked = False


    def output(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        video_paths = []
        logger.info("Merging images into GIF.")
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        output_dir = Path(f"{p.outpath_samples}/AnimateDiff/{date}")
        output_dir.mkdir(parents=True, exist_ok=True)
        step = params.video_length if params.video_length > params.batch_size else params.batch_size
        for i in range(res.index_of_first_image, len(res.images), step):
            # frame interpolation replaces video_list with interpolated frames
            # so make a copy instead of a slice (reference), to avoid modifying res
            frame_list = [image.copy() for image in res.images[i : i + params.video_length]]

            seq = images.get_next_sequence_number(output_dir, "")
            filename_suffix = f"-{params.request_id}" if params.request_id else ""
            filename = f"{seq:05}-{res.all_seeds[(i-res.index_of_first_image)]}{filename_suffix}"

            video_path_prefix = output_dir / filename

            frame_list = self._add_reverse(params, frame_list)
            frame_list = self._interp(p, params, frame_list, filename)
            video_paths += self._save(params, frame_list, video_path_prefix, res, i)

        if len(video_paths) > 0:
            if p.is_api:
                if not AnimateDiffOutput.api_encode_pil_to_base64_hooked:
                    # TODO: remove this hook when WebUI is updated to v1.7.0
                    logger.info("Hooking api.encode_pil_to_base64 to encode video to base64")
                    AnimateDiffOutput.api_encode_pil_to_base64_hooked = True
                    from modules.api import api
                    api_encode_pil_to_base64 = api.encode_pil_to_base64
                    def hooked_encode_pil_to_base64(image):
                        if isinstance(image, str):
                            return image
                        return api_encode_pil_to_base64(image)
                    api.encode_pil_to_base64 = hooked_encode_pil_to_base64
                res.images = self._encode_video_to_b64(video_paths) + (frame_list if 'Frame' in params.format else [])
            else:
                res.images = video_paths


    def _add_reverse(self, params: AnimateDiffProcess, frame_list: list):
        if params.video_length <= params.batch_size and params.closed_loop in ['A']:
            frame_list_reverse = frame_list[::-1]
            if len(frame_list_reverse) > 0:
                frame_list_reverse.pop(0)
            if len(frame_list_reverse) > 0:
                frame_list_reverse.pop(-1)
            return frame_list + frame_list_reverse
        return frame_list


    def _interp(
        self,
        p: StableDiffusionProcessing,
        params: AnimateDiffProcess,
        frame_list: list,
        filename: str
    ):
        if params.interp not in ['FILM']:
            return frame_list
        
        try:
            from deforum_helpers.frame_interpolation import (
                calculate_frames_to_add, check_and_download_film_model)
            from film_interpolation.film_inference import run_film_interp_infer
        except ImportError:
            logger.error("Deforum not found. Please install: https://github.com/deforum-art/deforum-for-automatic1111-webui.git")
            return frame_list

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

        film_in_between_frames_count = calculate_frames_to_add(len(frame_list), params.interp_x) 

        # save original frames to tmp folder for deforum input
        tmp_folder = f"{p.outpath_samples}/AnimateDiff/tmp"
        input_folder = f"{tmp_folder}/input"
        os.makedirs(input_folder, exist_ok=True)
        for tmp_seq, frame in enumerate(frame_list):
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
        frame_list = []
        for f in interp_frame_paths:
            with Image.open(f) as img:
                img.load()
                frame_list.append(img)
        
        # if saving PNG, enforce saving to custom folder
        if "PNG" in params.format:
            params.force_save_to_custom = True

        # remove tmp folder
        try: shutil.rmtree(tmp_folder)
        except OSError as e: print(f"Error: {e}")

        return frame_list


    def _save(
        self,
        params: AnimateDiffProcess,
        frame_list: list,
        video_path_prefix: Path,
        res: Processed,
        index: int,
    ):
        video_paths = []
        video_array = [np.array(v) for v in frame_list]
        infotext = res.infotexts[index]
        s3_enable =shared.opts.data.get("animatediff_s3_enable", False) 
        use_infotext = shared.opts.enable_pnginfo and infotext is not None
        if "PNG" in params.format and (shared.opts.data.get("animatediff_save_to_custom", False) or getattr(params, "force_save_to_custom", False)):
            video_path_prefix.mkdir(exist_ok=True, parents=True)
            for i, frame in enumerate(frame_list):
                png_filename = video_path_prefix/f"{i:05}.png"
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text('parameters', infotext)
                imageio.imwrite(png_filename, frame, pnginfo=png_info)

        if "GIF" in params.format:
            video_path_gif = str(video_path_prefix) + ".gif"
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
                            "scale": ("scale", f"{frame_list[0].width}:{frame_list[0].height}")
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
                                exif_tool.execute("-overwrite_original", f"-Comment={escaped_infotext}", video_path_gif)
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
            video_path_mp4 = str(video_path_prefix) + ".mp4"
            video_paths.append(video_path_mp4)
            try:
                import av
            except ImportError:
                from launch import run_pip
                run_pip(
                    "install pyav",
                    "sd-webui-animatediff MP4 save requirement: PyAV",
                )
                import av
            options = {
                "crf": str(shared.opts.data.get("animatediff_mp4_crf", 23))
            }
            preset = shared.opts.data.get("animatediff_mp4_preset", "")
            if preset != "": options["preset"] = preset
            tune = shared.opts.data.get("animatediff_mp4_tune", "")
            if tune != "": options["tune"] = tune
            output = av.open(video_path_mp4, "w")
            logger.info(f"Saving {video_path_mp4}")
            if use_infotext:
                output.metadata["Comment"] = infotext
            stream = output.add_stream('libx264', params.fps, options=options)
            stream.width = frame_list[0].width
            stream.height = frame_list[0].height
            for img in video_array:
                frame = av.VideoFrame.from_ndarray(img)
                packet = stream.encode(frame)
                output.mux(packet)
            packet = stream.encode(None)
            output.mux(packet)
            output.close()

        if "TXT" in params.format and res.images[index].info is not None:
            video_path_txt = str(video_path_prefix) + ".txt"
            with open(video_path_txt, "w", encoding="utf8") as file:
                file.write(f"{infotext}\n")

        if "WEBP" in params.format:
            if PIL.features.check('webp_anim'):            
                video_path_webp = str(video_path_prefix) + ".webp"
                video_paths.append(video_path_webp)
                exif_bytes = b''
                if use_infotext:
                    exif_bytes = piexif.dump({
                        "Exif":{
                            piexif.ExifIFD.UserComment:piexif.helper.UserComment.dump(infotext, encoding="unicode")
                        }})
                lossless = shared.opts.data.get("animatediff_webp_lossless", False)
                quality = shared.opts.data.get("animatediff_webp_quality", 80)
                logger.info(f"Saving {video_path_webp} with lossless={lossless} and quality={quality}")
                imageio.imwrite(video_path_webp, video_array, plugin='pillow',
                    duration=int(1 / params.fps * 1000), loop=params.loop_number,
                    lossless=lossless, quality=quality, exif=exif_bytes
                )
                # see additional Pillow WebP options at https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp
            else:
                logger.warn("WebP animation in Pillow requires system WebP library v0.5.0 or later")
        if "WEBM" in params.format:
            video_path_webm = str(video_path_prefix) + ".webm"
            video_paths.append(video_path_webm)
            logger.info(f"Saving {video_path_webm}")
            with imageio.imopen(video_path_webm, "w", plugin="pyav") as file:
                if use_infotext:
                    file.container_metadata["Title"] = infotext
                    file.container_metadata["Comment"] = infotext
                file.write(video_array, codec="vp9", fps=params.fps)
        
        if s3_enable:
            for video_path in video_paths: self._save_to_s3_stroge(video_path)  
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


    def _encode_video_to_b64(self, paths):
        videos = []
        for v_path in paths:
            with open(v_path, "rb") as video_file:
                videos.append(base64.b64encode(video_file.read()).decode("utf-8"))
        return videos

    def _install_requirement_if_absent(self,lib):
        import launch
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"animatediff requirement: {lib}")

    def _exist_bucket(self,s3_client,bucketname):
        try:
            s3_client.head_bucket(Bucket=bucketname)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise

    def _save_to_s3_stroge(self ,file_path):
        """
        put object to object storge
        :type bucketname: string
        :param bucketname: will save to this 'bucket' , access_key and secret_key must have permissions to save 
        :type file  : file 
        :param file : the local file 
        """        
        self._install_requirement_if_absent('boto3')
        import boto3
        from botocore.exceptions import ClientError
        import os
        host = shared.opts.data.get("animatediff_s3_host", '127.0.0.1')
        port = shared.opts.data.get("animatediff_s3_port", '9001') 
        access_key = shared.opts.data.get("animatediff_s3_access_key", '') 
        secret_key = shared.opts.data.get("animatediff_s3_secret_key", '') 
        bucket = shared.opts.data.get("animatediff_s3_storge_bucket", '') 
        client = boto3.client(
                service_name='s3',
                aws_access_key_id = access_key,
                aws_secret_access_key = secret_key,
                endpoint_url=f'http://{host}:{port}',
                )
                
        if not os.path.exists(file_path): return
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        if not self._exist_bucket(client,bucket):
            client.create_bucket(Bucket=bucket)

        filename = os.path.split(file_path)[1]
        targetpath = f"{date}/{filename}"
        client.upload_file(file_path, bucket,  targetpath)
        logger.info(f"{file_path} saved to s3 in bucket: {bucket}")
        return f"http://{host}:{port}/{bucket}/{targetpath}"
        
