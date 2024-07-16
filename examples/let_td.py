from numpy import array, pi, arange, sqrt
import matplotlib.pyplot as pp
from nsimplex.plot import plot_polygons, plot_polygons_list
from invtphase.config import load_primary_spectrometer
import scipp as sc
from nexus_interface import histogram_monitor, chopper_speed_phase
import scipp.constants as scc
from pathlib import Path

primary = load_primary_spectrometer('let.yaml')

nx_file = str(Path.home().joinpath('Documents', 'ISIS', 'LET', 'LET00076866.nxs'))
choppers = [c.name for _, c in primary.pairs]
speed, phase, ei = chopper_speed_phase(nx_file, choppers, 'Ei')

primary.set_frequencies([speed[x] for x in choppers])
# primary.set_phase_angles([speed_phase[f'{x}_phase'] for x in choppers])
# primary.set_delays([phase[x] for x in choppers])

ei = sc.scalar(ei, unit='meV')
vi = sc.sqrt(2 * ei / scc.m_n).to(unit='m/s')
primary.setup_phases(vi.value, True)


time_distance = primary.forward_time_distance_diagram()

# et = [sc.to_unit(v, 'sec').value for v in phase.values()]

fig, ax = pp.subplots(1, 1)
# target_poly.plot(ax, color='red')
ax.plot(*primary.time_distance_openings(0, 1/primary.source.frequency), '-k')
ax.plot(*primary.time_distance_frames(0, 1/primary.source.frequency), linestyle='--', color='gray')
# ax.plot(*primary.time_distance_frames(0, 1/primary.source.frequency, extra_times=et), color='red')
plot_polygons_list(ax, time_distance, alpha=1.0, color='green')

pp.show()
