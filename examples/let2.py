from pathlib import Path
from numpy import array, pi, arange, sqrt
import matplotlib.pyplot as pp
from figures import to_meV, figure_source_sample_detector
from invtphase import FlightPath, SecondarySpectrometer
from invtphase.config import load_primary_spectrometer
import scipp as sc
import scipp.constants as scc
from nexus_interface import histogram_monitor, chopper_speed_phase

primary = load_primary_spectrometer('let.yaml')

nx_file = Path.home().joinpath('Documents','ISIS','LET','LET00076866.nxs')
choppers = [f'Chopper{i:d}' for i in (1, 2, 3, 4, 5)]
choppers = [f'Chopper{i}' for i in ('1_Disk1', '1_Disk2', 2, 3, 4, '5_Disk1', '5_Disk2')]
speed, phase, ei = chopper_speed_phase(str(nx_file), choppers, 'Ei')

primary.set_frequencies([speed[x] for x in choppers])
# primary.set_phase_angles([speed_phase[f'{x}_phase'] for x in choppers])
# primary.set_delays([phase[x] for x in choppers])

ei = sc.scalar(ei, unit='meV')
vi = sc.sqrt(2 * ei / scc.m_n).to(unit='m/s')
primary.setup_phases(vi.value, True)
#
# on_source_alt = primary.project_on_source_alternate()
# on_sample_alt = primary.project_on_sample_alternate()

# src = primary.project_all_on_source()
proj, lproj = primary.project_transmitted_on_source()

samp, lsamp = primary.project_transmitted_on_sample()

fig, axs = pp.subplots(2,2)

from nsimplex.plot import plot_polygons, plot_polygons_list
plot_polygons_list(axs[0,0], lproj)
plot_polygons(axs[0,1], proj)
plot_polygons_list(axs[1,0], lsamp)
plot_polygons(axs[1,1], samp)

fig2, ax2 = pp.subplots(len(lproj), 2)
for ax, lp in zip(ax2[:,0], lproj):
    plot_polygons(ax, lp)
for ax, ls in zip(ax2[:,1], lsamp):
    plot_polygons(ax, ls)

pp.show()
