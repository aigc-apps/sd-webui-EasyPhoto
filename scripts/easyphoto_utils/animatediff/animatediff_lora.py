import os
import re
import sys

from modules import sd_models, shared
from modules.paths import extensions_builtin_dir

from .animatediff_logger import logger_animatediff as logger

sys.path.append(f"{extensions_builtin_dir}/Lora")

class AnimateDiffLora:
    original_load_network = None

    def __init__(self, v2: bool):
        self.v2 = v2

    def hack(self):
        if not self.v2:
            return

        if AnimateDiffLora.original_load_network is not None:
            logger.info("AnimateDiff LoRA already hacked")
            return

        logger.info("Hacking LoRA module to support motion LoRA")
        import network
        import networks
        AnimateDiffLora.original_load_network = networks.load_network
        original_load_network = AnimateDiffLora.original_load_network

        def mm_load_network(name, network_on_disk):

            def convert_mm_name_to_compvis(key):
                sd_module_key, _, network_part = re.split(r'(_lora\.)', key)
                sd_module_key = sd_module_key.replace("processor.", "").replace("to_out", "to_out.0")
                return sd_module_key, 'lora_' + network_part

            net = network.Network(name, network_on_disk)
            net.mtime = os.path.getmtime(network_on_disk.filename)

            sd = sd_models.read_state_dict(network_on_disk.filename)
            
            if 'motion_modules' in list(sd.keys())[0]:
                logger.info(f"Loading motion LoRA {name} from {network_on_disk.filename}")
                matched_networks = {}

                for key_network, weight in sd.items():
                    key, network_part = convert_mm_name_to_compvis(key_network)
                    sd_module = shared.sd_model.network_layer_mapping.get(key, None)

                    assert sd_module is not None, f"Failed to find sd module for key {key}."

                    if key not in matched_networks:
                        matched_networks[key] = network.NetworkWeights(
                            network_key=key_network, sd_key=key, w={}, sd_module=sd_module)

                    matched_networks[key].w[network_part] = weight

                for key, weights in matched_networks.items():
                    net_module = networks.module_types[0].create_module(net, weights)
                    assert net_module is not None, "Failed to create motion module LoRA"
                    net.modules[key] = net_module

                return net
            else:
                del sd
                return original_load_network(name, network_on_disk)

        networks.load_network = mm_load_network

    
    def restore(self):
        if not self.v2:
            return

        if AnimateDiffLora.original_load_network is None:
            logger.info("AnimateDiff LoRA already restored")
            return

        logger.info("Restoring hacked LoRA")
        import networks
        networks.load_network = AnimateDiffLora.original_load_network
        AnimateDiffLora.original_load_network = None
