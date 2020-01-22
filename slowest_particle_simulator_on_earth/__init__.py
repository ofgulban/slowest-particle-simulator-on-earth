"""For having the version."""

import pkg_resources

__version__ = pkg_resources.require(
    "slowest_particle_simulator_on_earth")[0].version
