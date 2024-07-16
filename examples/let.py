from numpy import array, pi, arange, sqrt
import matplotlib.pyplot as pp
from figures import to_meV, figure_source_sample_detector
from invtphase import FlightPath, SecondarySpectrometer
from invtphase.config import load_primary_spectrometer
import scipp as sc
import scipp.constants as scc

from nexus_interface import histogram_monitor, chopper_speed_phase

primary = load_primary_spectrometer('let.yaml')

# Read chopper speeds and phases from a NeXus file's logs:
from pathlib import Path
# nx_file = 'LET00011130.nxs'
# nx_file = 'LET00058906.nxs'
# nx_file = 'LET00078202.nxs'
nx_file = 'LET00076866.nxs'
nx_file = str(Path.home().joinpath('Documents', 'ISIS', 'LET', nx_file))

choppers = [c.name for _, c in primary.pairs]
speed, phase, target_ei = chopper_speed_phase(nx_file, choppers, 'Ei')


primary.set_frequencies([speed[x] for x in choppers])
# primary.set_phase_angles([speed_phase[f'{x}_phase'] for x in choppers])
# primary.set_delays([phase[x] for x in choppers])

ei = sc.scalar(target_ei, unit='meV')
vi = sc.sqrt(2 * ei / scc.m_n).to(unit='m/s')
primary.setup_phases(vi.value, True)


on_source_alt = primary.project_on_source_alternate()
on_sample_alt = primary.project_on_sample_alternate()

# Set up the secondary spectrometer
# (but only one detector for now)
slowest = primary.source.slowest
fastest = 100 / min([x.min(1) for x in on_sample_alt])
detector = FlightPath(name='Sample to Detector', velocity=(slowest, fastest), nominal=(3.5, 3.5))
cspec_secondary = SecondarySpectrometer([detector])

after_sample = cspec_secondary.sample_scatter(on_sample_alt)
on_detector = cspec_secondary.project_on_detectors(on_sample_alt)

source_meV = to_meV(on_source_alt)
sample_meV = to_meV(on_sample_alt)
# after_meV = to_meV(after_sample)
detector_meV = [to_meV(x) for x in on_detector]
figure_source_sample_detector(source_meV, sample_meV, detector_meV, plottype='energy', repeat=1/primary.source.frequency)

# figure_source_sample_detector(on_source_alt, on_sample_alt, after_sample, on_detector, repeat=1/primary.source.frequency)



pp.show()

for x in sample_meV:
    tavg, eavg = x.centroid
    tmin = x.min(0)
    print(f'{eavg: 8.3f} meV <{tavg:0.6f}> s |{tmin:0.6f}- s')



