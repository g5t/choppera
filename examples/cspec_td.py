from numpy import array, pi, arange, sqrt
import matplotlib.pyplot as pp
from scripts.nsimplex.plot import plot_polygons, plot_polygons_list
from invtphase.config import load_primary_spectrometer

primary = load_primary_spectrometer('cspec.yaml')
# we really need a better way for setting choppers
#
# pulse shaping [1, 12] x 14 Hz
primary.pairs[3][1].frequency = 2 * primary.source.frequency  # (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
# monochromator [2, 2, 24] x 14 Hz (always twice pulse shaping)
primary.pairs[5][1].frequency = 2 * primary.pairs[3][1].frequency
# pulser -- a rational fraction of the mono, which is also an integer multiple of the source
#  2: 1
#  4: 1, 2
#  6: 1, 3
#  8: 1, 2, 4
# 10: 1, 2, 5
# 12: 1, 2, 3, 4, 6
# 14: 1, 2, 7
# 16: 1, 2, 4, 8
# 18: 1, 2, 9
# 20: 1, 2, 4, 5, 10
# 22: 1, 11
# 24: 1, 2, 3, 4, 6, 12
primary.pairs[4][1].frequency = 4 * primary.source.frequency

for _, c in primary.pairs:
    print(f"{c}: {c.frequency}")

primary.setup_phases(500, True)  # target 1000 m / s neutrons

time_distance = primary.forward_time_distance_diagram()

fig, ax = pp.subplots(1, 1)
# target_poly.plot(ax, color='red')
ax.plot(*primary.time_distance_openings(0, 7/primary.source.frequency), '-k')
ax.plot(*primary.time_distance_frames(0, 7/primary.source.frequency), linestyle='--', color='gray')
plot_polygons_list(ax, time_distance, alpha=0.2)

pp.show()