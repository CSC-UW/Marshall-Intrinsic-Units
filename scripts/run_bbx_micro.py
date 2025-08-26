import argparse

import marshall_intrinsic_units

parser = argparse.ArgumentParser()
parser.add_argument("size", help="Subsystem size", type=int)
parser.add_argument("--isel", help="Subsystem index", type=int, default=None)
args = parser.parse_args()

marshall_intrinsic_units.run_blackbox_micro_example(
    subsystem_size=args.size, subsystem_index=args.isel
)
