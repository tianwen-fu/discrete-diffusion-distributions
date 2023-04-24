from copy import deepcopy

from .defaults import absorbing_default

__all__ = ["absorbing_center_M1", "absorbing_center_M5"]

# write 2 configs to avoid OOM
absorbing_center_M1 = {}
absorbing_center_M5 = {}
# M = 1
for num_classes in (50, 100, 200):
    aligned = deepcopy(absorbing_default)
    aligned.transition_kwargs["stationary_state"] = num_classes // 2
    aligned.num_classes = num_classes
    absorbing_center_M1[f"N{num_classes}_M1_aligned"] = aligned
    misaligned = deepcopy(absorbing_default)
    misaligned.transition_kwargs["stationary_state"] = 0
    misaligned.num_classes = num_classes
    absorbing_center_M1[f"N{num_classes}_M1_misaligned"] = misaligned

# M = 5
modes_base = deepcopy(absorbing_default)
modes_base.n_modes = 5
for num_classes in (50, 100, 200):
    align_center = num_classes // (5 + 1)
    misalign_center = num_classes - 1
    aligned = deepcopy(modes_base)
    aligned.transition_kwargs["stationary_state"] = align_center
    aligned.num_classes = num_classes
    absorbing_center_M5[f"N{num_classes}_M5_aligned"] = aligned
    misaligned = deepcopy(modes_base)
    misaligned.transition_kwargs["stationary_state"] = misalign_center
    misaligned.num_classes = num_classes
    absorbing_center_M5[f"N{num_classes}_M5_misaligned"] = misaligned
