"""Main entry point."""

import argparse
import slowest_particle_simulator_on_earth.config as cfg
from slowest_particle_simulator_on_earth.config import core, __version__


def main():
    """Commandline interface."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'filename',  metavar='path', nargs='+',
        help="Path to image. Multiple paths can be provided."
        )
    parser.add_argument(
        '--iterations', type=str, required=False,
        metavar=cfg.iterations, default=cfg.iterations,
        help="Number of iterations. Equal to number of frames generated."
        )
    parser.add_argument(
        '--explosiveness', type=str, required=False,
        metavar=cfg.explosiveness, default=cfg.explosiveness,
        help="Larger numbers for larger explosions."
        )

    args = parser.parse_args()
    cfg.iterations = args.iterations
    cfg.explosiveness = args.explosiveness

    # Welcome message
    welcome_str = '{} {}'.format(
        'Slowest particle simulator on earth', __version__)
    welcome_decor = '=' * len(welcome_str)
    print('{}\n{}\n{}'.format(welcome_decor, welcome_str, welcome_decor))

    # =========================================================================
    # TODO: Do simulation here

    # =========================================================================

    print('Finished.')


if __name__ == "__main__":
    main()
