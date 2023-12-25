import _thread
import argparse
import contextlib
import datetime
import heapq
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from diffusers.utils import is_wandb_available
from PIL import Image
from safetensors.torch import load_file

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
import utils.lora_utils as network_module
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.stat_tracking import PerPromptStatTracker


class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=""):
    """The context manager to limit the execution time of a function call given `seconds`.
    Borrowed from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call.
    """
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="The run name (needed to be unique) for wandb logging and checkpoint saving "
        "if not provided, will be auto-generated based on the datetime.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="The top-level logging directory for checkpoint saving.",
    )
    parser.add_argument(
        "--cache_log_file",
        type=str,
        default="train_kohya_log.txt",
        help="The output log file path. Use the same log file as train_lora.py",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train for. Each epoch is one round of sampling from the model " "followed by training on those samples.",
    )
    parser.add_argument("--save_freq", type=int, default=20, help="Number of epochs between saving model checkpoints.")
    parser.add_argument(
        "--num_checkpoint_limit",
        type=int,
        default=5,
        help="Number of checkpoints to keep before overwriting old ones.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), "
        "or a directory containing checkpoints, in which case the latest one will be used. "
        "`args.use_lora` must be set to the same value as the run that generated the saved checkpoint.",
    )

    # Sampling
    parser.add_argument(
        "--sample_num_steps",
        type=int,
        default=50,
        help="Number of sampler inference steps.",
    )
    parser.add_argument("--sample_guidance_scale", type=int, default=5, help="A guidance_scale during training for sampling.")
    parser.add_argument(
        "--sample_eta",
        type=float,
        default=1.0,
        help="The amount of noise injected into the DDIM sampling process.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="The batch size (per GPU!) to use for sampling.",
    )
    parser.add_argument(
        "--sample_num_batches_per_epoch",
        type=int,
        default=2,
        help="Number of batches to sample per epoch. The total number of samples per epoch is "
        "`sample_num_batches_per_epoch * sample_batch_size * num_gpus.`",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. (SD base model config.)",
    )
    parser.add_argument(
        "--pretrained_model_ckpt",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--original_config",
        type=str,
        required=True,
        help="Path to .yaml config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--face_lora_path",
        type=str,
        required=True,
        help="Path to the face lora model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="whether or not to use LoRA.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Number of epochs to train for. Each epoch is one round of sampling from the model " "followed by training on those samples.",
    )
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_inner_epochs",
        type=int,
        default=1,
        help="Number of inner epochs per outer epoch. Each inner epoch is one iteration "
        "through the data collected during one outer epoch's round of sampling.",
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        help="Whether or not to use classifier-free guidance during training. if enabled, the same guidance "
        "scale used during sampling will be used during training.",
    )
    parser.add_argument(
        "--adv_clip_max",
        type=float,
        default=5.0,
        help="Clip advantages to the range [-train_adv_clip_max, train_adv_clip_max].",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-4,
        help="the PPO clip range.",
    )
    parser.add_argument(
        "--timestep_fraction",
        type=float,
        default=1.0,
        help="The fraction of timesteps to train on. If set to less than 1.0, the model will be trained on a subset of the "
        "timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.",
    )

    # Prompt Function and Reward Function
    parser.add_argument(
        "--prompt_fn",
        type=str,
        default="easyphoto",
        help="The prompt function to use.",
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="faceid_v0",
        help="The reward function to use.",
    )
    parser.add_argument(
        "--target_image_dir",
        type=str,
        required=True,
        help="target_image_dir.",
    )

    # Per-Prompt Stat Tracking
    parser.add_argument(
        "--per_prompt_stat_tracking",
        action="store_true",
        help="Number of reward values to store in the buffer for each prompt. The buffer persists across epochs.",
    )
    parser.add_argument(
        "--per_prompt_stat_tracking_buffer_size",
        type=float,
        default=32,
        help="Number of reward values to store in the buffer for each prompt. The buffer persists across epochs.",
    )
    parser.add_argument(
        "--per_prompt_stat_tracking_min_count",
        type=int,
        default=16,
        help="The minimum number of reward values to store in the buffer before using the per-prompt mean and std. "
        "If the buffer contains fewer than `min_count` values, the mean and std of the entire batch will be used.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`' ' (default) and `"wandb"`.'),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id

    if args.resume_from:
        args.resume_from = os.path.normpath(os.path.expanduser(args.resume_from))
        if "checkpoint_" not in os.path.basename(args.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(args.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {args.resume_from}")
            args.resume_from = os.path.join(
                args.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(args.sample_num_steps * args.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=args.logdir,
        automatic_checkpoint_naming=True,
        total_limit=args.num_checkpoint_limit,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    accelerator = Accelerator(
        log_with=args.report_to,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want args.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=args.gradient_accumulation_steps * num_train_timesteps,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.logdir is not None:
            os.makedirs(args.logdir, exist_ok=True)

    if accelerator.is_main_process:
        if args.report_to == "wandb":
            accelerator.init_trackers(
                project_name="EasyPhoto-ddpo-pytorch", config=vars(args), init_kwargs={"wandb": {"dir": args.logdir, "name": args.run_name}}
            )
        else:
            accelerator.init_trackers("EasyPhoto-ddpo-pytorch", config=vars(args))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(args.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = download_from_original_stable_diffusion_ckpt(
        args.pretrained_model_ckpt,
        original_config_file=args.original_config,
        pipeline_class=StableDiffusionPipeline,
        model_type=None,
        stable_unclip=None,
        controlnet=False,
        from_safetensors=True,
        extract_ema=False,
        image_size=None,
        scheduler_type="pndm",
        num_in_channels=None,
        upcast_attention=None,
        load_safety_checker=False,
        prediction_type=None,
        text_encoder=None,
        tokenizer=None,
    )
    pipeline.scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    face_lora_state_dict = load_file(args.face_lora_path, device="cpu")
    network_module.merge_lora(pipeline, face_lora_state_dict)

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not args.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if args.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if args.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.rank)
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    reward_mean_list, reward_std_list = [], []
    cur_best_reward_mean, reward_mean_heap = (float("-inf"), ""), []

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if args.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not args.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if args.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained.model, revision=args.pretrained.revision, subfolder="unet", use_safetensors=False
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not args.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", use_safetensors=False)
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # prepare prompt and reward fn
    if hasattr(ddpo_pytorch.prompts, args.prompt_fn):
        prompt_fn = getattr(ddpo_pytorch.prompts, args.prompt_fn)
    else:
        raise ValueError(
            "Prompt function {} is not defined in {}/ddpo_pytorch/prompts.py." "".format(args.prompt_fn, os.path.abspath(__file__))
        )
    if hasattr(ddpo_pytorch.rewards, args.reward_fn):
        reward_fn = getattr(ddpo_pytorch.rewards, args.reward_fn)(target_image_dir=args.target_image_dir)
    else:
        raise ValueError(
            "Reward function {} is not defined in {}/ddpo_pytorch/rewards.py" "".format(args.reward_fn, os.path.abspath(__file__))
        )

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.sample_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size, 1, 1)

    # initialize stat tracker
    if args.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            args.per_prompt_stat_tracking_buffer_size,
            args.per_prompt_stat_tracking_min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if args.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # Train!
    samples_per_epoch = args.sample_batch_size * accelerator.num_processes * args.sample_num_batches_per_epoch
    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Sample batch size per device = {args.sample_batch_size}")
    print(f"  Train batch size per device = {args.train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print("")
    print(f"  Total number of samples per epoch = {samples_per_epoch}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    print(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    print(f"  Number of inner epochs = {args.num_inner_epochs}")

    assert args.sample_batch_size >= args.train_batch_size
    assert args.sample_batch_size % args.train_batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        accelerator.load_state(args.resume_from)
        first_epoch = int(args.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    user_id = os.path.basename(os.path.dirname(args.logdir))
    # check log path
    if accelerator.is_main_process:
        output_log = open(args.cache_log_file, "w")

    global_step = 0
    for epoch in range(first_epoch, args.num_epochs):
        # SAMPLING
        pipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(args.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts = [prompt_fn() for _ in range(args.sample_batch_size)]

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=args.sample_num_steps,
                    guidance_scale=args.sample_guidance_scale,
                    eta=args.sample_eta,
                    output_type="pt",
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(args.sample_batch_size, 1)  # (batch_size, num_steps)

            rewards = reward_fn(images, prompts)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards = sample["rewards"]
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {"reward": rewards, "epoch": epoch, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )
        time_info = str(time.asctime(time.localtime(time.time())))
        log_line = (
            "{}: reinforcement learning of {} at step {}. The mean and std of face similarity score "
            "is {:.4f} and {:.4f}.\n".format(time_info, user_id, global_step, rewards.mean(), rewards.std())
        )
        reward_mean_list.append(rewards.mean())
        reward_std_list.append(rewards.std())
        if accelerator.is_main_process:
            output_log.write(log_line)
            output_log.flush()

        # Tensorboard does not support adding caption for image data.
        if is_wandb_available() and args.report_to == "wandb":
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            # Save more disk memory.
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                accelerator.log(
                    {
                        "images": [
                            wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt} | {reward:.2f}")
                            for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                        ],
                    },
                    step=global_step,
                )

        # per-prompt mean/std tracking
        if args.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages).reshape(accelerator.num_processes, -1)[accelerator.process_index].to(accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == args.sample_batch_size * args.sample_num_batches_per_epoch
        assert num_timesteps == args.sample_num_steps

        # TRAINING
        for inner_epoch in range(args.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, args.train_batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if args.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(pipeline.unet):
                        with autocast():
                            if args.cfg:
                                noise_pred = pipeline.unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + args.sample_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            else:
                                noise_pred = pipeline.unet(sample["latents"][:, j], sample["timesteps"][:, j], embeds).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=args.sample_eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(sample["advantages"], -args.adv_clip_max, args.adv_clip_max)
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > args.clip_range).float()))
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(trainable_layers.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (i + 1) % args.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        # Save the best checkpoint and maintain a heap (except for epoch 0) with length `num_checkpoint_limit - 1`
        # by `reward_mean`. Note that, `reward_mean` corresponds to the model saved in last epoch.
        if epoch % args.save_freq == 0 and accelerator.is_main_process:
            if reward_mean_list[-1] >= cur_best_reward_mean[0]:
                # (reward_mean, -1) => baseline will not be copied to best_outputs.
                cur_best_reward_mean = (reward_mean_list[-1], accelerator.save_iteration - 1)
                best_ckpt_src_dir = os.path.join(args.logdir, "checkpoints", "checkpoint_{}".format(accelerator.save_iteration - 1))
                if os.path.exists(best_ckpt_src_dir):
                    best_ckpt_dst_dir = os.path.join(args.logdir, "best_outputs")
                    if os.path.exists(best_ckpt_dst_dir):
                        shutil.rmtree(best_ckpt_dst_dir)
                    shutil.copytree(best_ckpt_src_dir, best_ckpt_dst_dir)
                    print(
                        "Copy the checkpoint directory: {} with the highest reward {} to best_outputs".format(
                            best_ckpt_src_dir, cur_best_reward_mean
                        )
                    )
            if len(reward_mean_heap) < args.num_checkpoint_limit - 1:
                if accelerator.save_iteration > 0:
                    heapq.heappush(reward_mean_heap, (reward_mean_list[-1], accelerator.save_iteration - 1))
            else:
                reward_save_iteration = (reward_mean_list[-1], accelerator.save_iteration - 1)
                if reward_mean_list[-1] >= reward_mean_heap[0][0]:
                    reward, save_iteration = heapq.heappushpop(reward_mean_heap, reward_save_iteration)
                    popped_ckpt_dir = os.path.join(args.logdir, "checkpoints", "checkpoint_{}".format(save_iteration))
                    if os.path.exists(popped_ckpt_dir):
                        shutil.rmtree(popped_ckpt_dir)
                        print("Delete the checkpoint directory: {} with the smallest reward {} in the heap".format(popped_ckpt_dir, reward))
                else:
                    last_ckpt_dir = os.path.join(args.logdir, "checkpoints", "checkpoint_{}".format(accelerator.save_iteration - 1))
                    shutil.rmtree(last_ckpt_dir)
                    print("Delete last checkpoint directory: {} with smaller reward {}".format(last_ckpt_dir, reward_mean_list[-1]))
            accelerator.save_state()
            np.savetxt(os.path.join(args.logdir, "reward_mean.txt"), np.array(reward_mean_list), delimiter=",", fmt="%.4f")
            np.savetxt(os.path.join(args.logdir, "reward_std.txt"), np.array(reward_std_list), delimiter=",", fmt="%.4f")

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # accelerator.save_model(trainable_layers, safetensor_save_path, safe_serialization=True)
        # we will remove cache_log_file after train
        with open(args.cache_log_file, "w") as _:
            pass
    accelerator.end_training()


if __name__ == "__main__":
    if "MAX_RL_TIME" in os.environ:
        MAX_RL_TIME = int(os.getenv("MAX_RL_TIME"))
        try:
            with time_limit(MAX_RL_TIME):
                main()
        except TimeoutException:
            print("Reinforcement learning timed out after {}!".format(MAX_RL_TIME))
    else:
        main()
