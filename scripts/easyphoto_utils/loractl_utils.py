import os
import re
import sys

import numpy as np
from modules import shared, extra_networks
from scripts.easyphoto_config import data_path

Lora_extensions_path = os.path.join(data_path, "extensions", "Lora")
Lora_extensions_builtin_path = os.path.join(data_path, "extensions-builtin", "Lora")


if os.path.exists(Lora_extensions_path):
    lora_path = Lora_extensions_path
    ctl_lora_flag = True
elif os.path.exists(Lora_extensions_builtin_path):
    lora_path = Lora_extensions_builtin_path
    ctl_lora_flag = True
else:
    ctl_lora_flag = False
    LoraCtlNetwork = None
    apply = None
    set_active = None
    reset_lora_weights = None
    set_hire_fix = None
    ctl_lora_flag = None


if ctl_lora_flag:
    sys.path.insert(0, lora_path)
    import extra_networks_lora
    import network

    sys.path.remove(lora_path)

    hire_fix_flag = False
    loractl_active = True
    lora_weights = {}

    def is_hire_fix():
        return hire_fix_flag

    def set_hire_fix(value):
        global hire_fix_flag
        hire_fix_flag = value

    def is_active():
        global loractl_active
        return loractl_active

    def set_active(value):
        global loractl_active
        loractl_active = value

    def reset_lora_weights():
        global lora_weights
        lora_weights.clear()

    # Given a string like x@y,z@a, returns [[x, z], [y, a]] sorted for consumption by np.interp
    def sorted_positions(raw_steps):
        steps = [[float(s.strip()) for s in re.split("[@~]", x)] for x in re.split("[,;]", str(raw_steps))]
        # If we just got a single number, just return it
        if len(steps[0]) == 1:
            return steps[0][0]

        # Add implicit 1s to any steps which don't have a weight
        steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps]

        # Sort by index
        steps.sort(key=lambda k: k[1])

        steps = [list(v) for v in zip(*steps)]
        return steps

    def params_to_weights(params):
        weights = {"unet": None, "te": 1.0, "hrunet": None, "hrte": None}

        if len(params.positional) > 1:
            weights["te"] = sorted_positions(params.positional[1])

        if len(params.positional) > 2:
            weights["unet"] = sorted_positions(params.positional[2])

        if params.named.get("te"):
            weights["te"] = sorted_positions(params.named.get("te"))

        if params.named.get("unet"):
            weights["unet"] = sorted_positions(params.named.get("unet"))

        if params.named.get("hr"):
            weights["hrunet"] = sorted_positions(params.named.get("hr"))
            weights["hrte"] = sorted_positions(params.named.get("hr"))

        if params.named.get("hrunet"):
            weights["hrunet"] = sorted_positions(params.named.get("hrunet"))

        if params.named.get("hrte"):
            weights["hrte"] = sorted_positions(params.named.get("hrte"))

        # If unet ended up unset, then use the te value
        weights["unet"] = weights["unet"] if weights["unet"] is not None else weights["te"]
        # If hrunet ended up unset, use unet value
        weights["hrunet"] = weights["hrunet"] if weights["hrunet"] is not None else weights["unet"]
        # If hrte ended up unset, use te value
        weights["hrte"] = weights["hrte"] if weights["hrte"] is not None else weights["te"]

        return weights

    class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
        # Hijack the params parser and feed it dummy weights instead so it doesn't choke trying to
        # parse our extended syntax
        def activate(self, p, params_list):
            if not is_active():
                return super().activate(p, params_list)

            # list params
            for params in params_list:
                assert params.items
                name = params.positional[0]
                if lora_weights.get(name, None) is None:
                    lora_weights[name] = params_to_weights(params)
                # The hardcoded 1 weight is fine here, since our actual patch looks up the weights from
                # our lora_weights dict
                params.positional = [name, 1]
                params.named = {}
            return super().activate(p, params_list)

    def calculate_weight(m, step, max_steps, step_offset=2):
        if isinstance(m, list):
            # normalize the step to 0~1
            if m[1][-1] <= 1.0:
                if max_steps > 0:
                    step = (step) / (max_steps - step_offset)
                else:
                    step = 1.0
            else:
                step = step
            # get value from interp between m[1]~m[0]
            v = np.interp(step, m[1], m[0])
            return v
        else:
            return m

    def get_weight(m):
        return calculate_weight(m, shared.state.sampling_step, shared.state.sampling_steps, step_offset=2)

    def get_dynamic_te(self):
        if self.name in lora_weights:
            key = "te" if not is_hire_fix() else "hrte"
            w = lora_weights[self.name]
            return get_weight(w.get(key, self._te_multiplier))

        return get_weight(self._te_multiplier)

    def get_dynamic_unet(self):
        if self.name in lora_weights:
            key = "unet" if not is_hire_fix() else "hrunet"
            w = lora_weights[self.name]
            return get_weight(w.get(key, self._unet_multiplier))

        return get_weight(self._unet_multiplier)

    def set_dynamic_te(self, value):
        self._te_multiplier = value

    def set_dynamic_unet(self, value):
        self._unet_multiplier = value

    def apply():
        if getattr(network.Network, "te_multiplier", None) is None:
            network.Network.te_multiplier = property(get_dynamic_te, set_dynamic_te)
            network.Network.unet_multiplier = property(get_dynamic_unet, set_dynamic_unet)
