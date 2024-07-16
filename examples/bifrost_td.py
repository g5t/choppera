"""
BIFROST chopper specifications from CHESS document ESS-3164400 Rev. 1 page 12
Plus updated information from Rasmus on 2022-02-04:
    FOC1: 38.40 deg
    FOC2: 52.14 deg
    PSC distance: 6342 mm from the W4 ISCS origo
    FOC 1 distance: 8530 mm from the ISCS origo
    FOC 2 distance: 14973 mm from the ISCS origo
    BW distance: 78000 mm from the ISCS origo
"""
from numpy import array
import matplotlib.pyplot as pp
from scripts.nsimplex.plot import plot_polygons, plot_polygons_list
from nsimplex import Polygon, Border
from invtphase.flightpaths import NEUTRON_MASS_OVER_PLANCK_CONSTANT

from invtphase.config import load_primary_spectrometer

primary = load_primary_spectrometer('bifrost.yaml')
# we need a better way for setting chopper frequencies ...
primary.pairs[0][1].frequency = 15 * primary.source.frequency

target_wavelength = 3.3  # ångstrom
target_velocity = 1 / (NEUTRON_MASS_OVER_PLANCK_CONSTANT * target_wavelength)  # m / s, targeted neutron velocity
target_time = 162 / target_velocity
target_path = array([[0, 0], [primary.source.duration, 0], [primary.source.duration + target_time, 162], [target_time, 162]])
target_poly = Polygon(Border(target_path), [])

primary.setup_phases(target_velocity, True)

time_distance = primary.forward_time_distance_diagram()

fig, ax = pp.subplots(1, 1)
# target_poly.plot(ax, color='red')
ax.plot(*primary.time_distance_openings(0, 3/primary.source.frequency), '-k')
plot_polygons_list(ax, time_distance, alpha=0.2)

pp.show()