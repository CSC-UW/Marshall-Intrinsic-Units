from pyphi import config

import marshall_intrinsic_units

with config.override(VALIDATE_SUBSYSTEM_STATES=False):
    marshall_intrinsic_units.run_binary_units_micro_example()
    marshall_intrinsic_units.summarize_binary_units_micro_example()
