import re
import torch

from modules.processing import StableDiffusionProcessing, Processed

from .animatediff_logger import logger_animatediff as logger
from .animatediff_infotext import write_params_txt


class AnimateDiffPromptSchedule:

    def __init__(self):
        self.prompt_map = None
        self.original_prompt = None


    def save_infotext_img(self, p: StableDiffusionProcessing):
        if self.prompt_map is not None:
            p.prompts = [self.original_prompt for _ in range(p.batch_size)]


    def save_infotext_txt(self, res: Processed):
        if self.prompt_map is not None:
            parts = res.info.split('\nNegative prompt: ', 1)
            if len(parts) > 1:
                res.info = f"{self.original_prompt}\nNegative prompt: {parts[1]}"
                for i in range(len(res.infotexts)):
                    parts = res.infotexts[i].split('\nNegative prompt: ', 1)
                    if len(parts) > 1:
                        res.infotexts[i] = f"{self.original_prompt}\nNegative prompt: {parts[1]}"
                write_params_txt(res.info)


    def parse_prompt(self, p: StableDiffusionProcessing):
        if type(p.prompt) is not str:
            logger.warn("prompt is not str, cannot support prompt map")
            return

        lines = p.prompt.strip().split('\n')
        data = {
            'head_prompts': [],
            'mapp_prompts': {},
            'tail_prompts': []
        }

        mode = 'head'
        for line in lines:
            if mode == 'head':
                if re.match(r'^\d+:', line):
                    mode = 'mapp'
                else:
                    data['head_prompts'].append(line)
                    
            if mode == 'mapp':
                match = re.match(r'^(\d+): (.+)$', line)
                if match:
                    frame, prompt = match.groups()
                    data['mapp_prompts'][int(frame)] = prompt
                else:
                    mode = 'tail'
                    
            if mode == 'tail':
                data['tail_prompts'].append(line)
        
        if data['mapp_prompts']:
            logger.info("You are using prompt travel.")
            self.prompt_map = {}
            prompt_list = []
            last_frame = 0
            current_prompt = ''
            for frame, prompt in data['mapp_prompts'].items():
                prompt_list += [current_prompt for _ in range(last_frame, frame)]
                last_frame = frame
                current_prompt = f"{', '.join(data['head_prompts'])}, {prompt}, {', '.join(data['tail_prompts'])}"
                self.prompt_map[frame] = current_prompt
            prompt_list += [current_prompt for _ in range(last_frame, p.batch_size)]
            assert len(prompt_list) == p.batch_size, f"prompt_list length {len(prompt_list)} != batch_size {p.batch_size}"
            self.original_prompt = p.prompt
            p.prompt = prompt_list * p.n_iter


    def single_cond(self, center_frame, video_length: int, cond: torch.Tensor, closed_loop = False):
        if closed_loop:
            key_prev = list(self.prompt_map.keys())[-1]
            key_next = list(self.prompt_map.keys())[0]
        else:
            key_prev = list(self.prompt_map.keys())[0]
            key_next = list(self.prompt_map.keys())[-1]

        for p in self.prompt_map.keys():
            if p > center_frame:
                key_next = p
                break
            key_prev = p

        dist_prev = center_frame - key_prev
        if dist_prev < 0:
            dist_prev += video_length
        dist_next = key_next - center_frame
        if dist_next < 0:
            dist_next += video_length

        if key_prev == key_next or dist_prev + dist_next == 0:
            return cond[key_prev] if isinstance(cond, torch.Tensor) else {k: v[key_prev] for k, v in cond.items()}

        rate = dist_prev / (dist_prev + dist_next)
        if isinstance(cond, torch.Tensor):
            return AnimateDiffPromptSchedule.slerp(cond[key_prev], cond[key_next], rate)
        else: # isinstance(cond, dict)
            return {
                k: AnimateDiffPromptSchedule.slerp(v[key_prev], v[key_next], rate)
                for k, v in cond.items()
            }
    

    def multi_cond(self, cond: torch.Tensor, closed_loop = False):
        if self.prompt_map is None:
            return cond
        cond_list = [] if isinstance(cond, torch.Tensor) else {k: [] for k in cond.keys()}
        for i in range(cond.shape[0]):
            single_cond = self.single_cond(i, cond.shape[0], cond, closed_loop)
            if isinstance(cond, torch.Tensor):
                cond_list.append(single_cond)
            else:
                for k, v in single_cond.items():
                    cond_list[k].append(v)
        if isinstance(cond, torch.Tensor):
            return torch.stack(cond_list).to(cond.dtype).to(cond.device)
        else:
            return {k: torch.stack(v).to(cond[k].dtype).to(cond[k].device) for k, v in cond_list.items()}


    @staticmethod
    def slerp(
        v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
    ) -> torch.Tensor:
        u0 = v0 / v0.norm()
        u1 = v1 / v1.norm()
        dot = (u0 * u1).sum()
        if dot.abs() > DOT_THRESHOLD:
            return (1.0 - t) * v0 + t * v1
        omega = dot.acos()
        return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()
