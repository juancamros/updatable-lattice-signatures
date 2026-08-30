'''Visualiza en dos dimensiones el efecto del ruido gaussiano corrector.'''

import argparse

import numpy as np

from src.correccion_gaussiana import sample_update_demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--continuous",
        action="store_false",
        dest="round_samples",
        help="Genera muestras continuas en lugar de redondearlas a enteros.",
    )
    parser.set_defaults(round_samples=True)
    args = parser.parse_args(argv)

    delta = np.array([[-1, 0], [1, 1]])
    sample_update_demo(
        n_samples=args.samples,
        s_e=1.0,
        s_next=3.0,
        Delta=delta,
        seed=args.seed,
        round_samples=args.round_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
