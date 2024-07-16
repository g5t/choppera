from numpy import array, pi, arange, sqrt
import matplotlib.pyplot as pp
from figures import to_meV, figure_source_sample_detector
from invtphase import FlightPath, SecondarySpectrometer
from invtphase.config import load_primary_spectrometer

primary = load_primary_spectrometer('cspec.yaml')
# we really need a better way for setting choppers
# monochromator
primary.pairs[5][1].frequency = 24 * primary.source.frequency
# pulse shaping (always half of mono)
primary.pairs[3][1].frequency = primary.pairs[4][1].frequency / 2
# pulser -- a rational fraction of the mono, which is an integer multiple of the source ?
primary.pairs[4][1].frequency = 24 * primary.source.frequency

primary.setup_phases(500, True)  # target 1000 m / s neutrons

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