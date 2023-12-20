import argparse
import base64
import json
import os
import time
from datetime import datetime

import requests
from tqdm import tqdm


# Function to encode a video file to Base64
def encode_video_to_base64(video_file_path):
    with open(video_file_path, "rb") as video_file:
        # Read the video file as binary data
        video_data = video_file.read()
        # Encode the data to Base64
        video_base64 = base64.b64encode(video_data)
        return video_base64


# Function to decode Base64 encoded data and save it as a video file
def decode_base64_to_video(encoded_video, output_file_path):
    with open(output_file_path, "wb") as output_file:
        # Decode the Base64 encoded data
        video_data = base64.b64decode(encoded_video)
        # Write the decoded binary data to the file
        output_file.write(video_data)


def post(init_image, last_image=None, init_video=None, user_id=None, tabs=0, url="http://0.0.0.0:7860"):
    if user_id is None:
        user_id = "test"
    datas = json.dumps(
        {
            "user_ids": [user_id],
            "sd_model_checkpoint": "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
            "sd_model_checkpoint_for_animatediff_text2video": "majicmixRealistic_v7.safetensors",
            "sd_model_checkpoint_for_animatediff_image2video": "majicmixRealistic_v7.safetensors",
            "t2v_input_prompt": "1girl, (white hair, long hair), blue eyes, hair ornament, blue dress, standing, looking at viewer, shy, upper-body",
            "t2v_input_width": 512,
            "t2v_input_height": 768,
            "scene_id": "none",
            "upload_control_video": False,
            "upload_control_video_type": "openpose",
            "openpose_video": None,
            "init_image": init_image,
            "init_image_prompt": "",
            "last_image": last_image,
            "init_video": init_video,
            "additional_prompt": "masterpiece, beauty",
            "max_frames": 16,
            "max_fps": 8,
            "save_as": "gif",
            "first_diffusion_steps": 50,
            "first_denoising_strength": 0.45,
            "seed": -1,
            "crop_face_preprocess": True,
            "before_face_fusion_ratio": 0.5,
            "after_face_fusion_ratio": 0.5,
            "apply_face_fusion_before": True,
            "apply_face_fusion_after": True,
            "color_shift_middle": True,
            "super_resolution": True,
            "super_resolution_method": "gpen",
            "skin_retouching_bool": False,
            "makeup_transfer": False,
            "makeup_transfer_ratio": 0.50,
            "face_shape_match": False,
            "video_interpolation": False,
            "video_interpolation_ext": 1,
            "tabs": tabs,
            "ipa_control": False,
            "ipa_weight": 0.50,
            "ipa_image": None,
            "lcm_accelerate": False,
        }
    )
    r = requests.post(f"{url}/easyphoto/easyphoto_video_infer_forward", data=datas, timeout=1500)
    data = r.content.decode("utf-8")
    return data


if __name__ == "__main__":
    """
    There are two ways to test:
        The first: make sure the directory is full of readable images
        The second: public link of readable picture
    """
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument("--init_image_path", type=str, default="", help="Path to the init image path")
    parser.add_argument("--last_image_path", type=str, default="", help="Path to the last image path")
    parser.add_argument("--video_path", type=str, default="", help="Path to the video path")
    parser.add_argument("--output_path", type=str, default="./", help="Path to the output directory")
    parser.add_argument("--user_ids", type=str, default="test", help="Test user ids, split with space")

    args = parser.parse_args()

    init_image_path = args.init_image_path
    last_image_path = args.last_image_path
    video_path = args.video_path
    output_path = args.output_path
    user_ids = args.user_ids.split(" ")

    if output_path != "./":
        os.makedirs(output_path, exist_ok=True)

    # initiate time
    now_date = datetime.now()
    time_start = time.time()

    # -------------------test infer------------------- #
    # When there is no parameter input.
    if init_image_path == "" and last_image_path == "" and video_path == "":
        for user_id in tqdm(user_ids):
            outputs = post(None, None, None, user_id, tabs=0)
            outputs = json.loads(outputs)
            if outputs["output_video"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.mp4")
                image = decode_base64_to_video(outputs["output_video"], toutput_path)
            elif outputs["output_gif"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.gif")
                image = decode_base64_to_video(outputs["output_gif"], toutput_path)

    elif init_image_path != "" and last_image_path == "" and video_path == "":
        with open(init_image_path, "rb") as f:
            init_image = base64.b64encode(f.read()).decode("utf-8")

        for user_id in tqdm(user_ids):
            outputs = post(init_image, None, None, user_id, tabs=1)
            outputs = json.loads(outputs)
            if outputs["output_video"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.mp4")
                image = decode_base64_to_video(outputs["output_video"], toutput_path)
            elif outputs["output_gif"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.gif")
                image = decode_base64_to_video(outputs["output_gif"], toutput_path)

    elif init_image_path != "" and last_image_path != "" and video_path == "":
        with open(init_image_path, "rb") as f:
            init_image = base64.b64encode(f.read()).decode("utf-8")
        with open(last_image_path, "rb") as f:
            last_image = base64.b64encode(f.read()).decode("utf-8")

        for user_id in tqdm(user_ids):
            outputs = post(init_image, last_image, None, user_id, tabs=1)
            outputs = json.loads(outputs)
            if outputs["output_video"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.mp4")
                image = decode_base64_to_video(outputs["output_video"], toutput_path)
            elif outputs["output_gif"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.gif")
                image = decode_base64_to_video(outputs["output_gif"], toutput_path)

    elif init_image_path == "" and last_image_path == "" and video_path != "":
        with open(video_path, "rb") as f:
            init_video = base64.b64encode(f.read()).decode("utf-8")

        for user_id in tqdm(user_ids):
            outputs = post(None, None, init_video, user_id, tabs=2)
            outputs = json.loads(outputs)
            if outputs["output_video"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.mp4")
                image = decode_base64_to_video(outputs["output_video"], toutput_path)
            elif outputs["output_gif"] is not None:
                toutput_path = os.path.join(os.path.join(output_path), f"{user_id}_tmp.gif")
                image = decode_base64_to_video(outputs["output_gif"], toutput_path)

    # End of record time
    # The calculated time difference is the execution time of the program, expressed in seconds / s
    time_end = time.time()
    time_sum = time_end - time_start
    print("# --------------------------------------------------------- #")
    print(f"#   Total expenditure: {time_sum}s")
    print("# --------------------------------------------------------- #")
