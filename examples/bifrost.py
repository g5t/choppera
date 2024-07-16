from numpy import array, pi, arange, sqrt, arcsin
import matplotlib.pyplot as pp
from invtphase import SecondarySpectrometer, FlightPath, AnalyzerArm
from figures import to_meV, figure_source_sample, figure_source_sample_detector
from invtphase.config import load_primary_spectrometer
from nsimplex import Polygon, Border
from invtphase.flightpaths import NEUTRON_MASS_OVER_PLANCK_CONSTANT

primary = load_primary_spectrometer('bifrost.yaml')
# we need a better way for setting chopper frequencies ...
primary.pairs[0][1].frequency = 5 * primary.source.frequency

target_wavelength = 3.3  # ångstrom
target_velocity = 1 / (NEUTRON_MASS_OVER_PLANCK_CONSTANT * target_wavelength)  # m / s, targeted neutron velocity
target_time = 162 / target_velocity
target_path = array([[0, 0], [primary.source.duration, 0], [primary.source.duration + target_time, 162], [target_time, 162]])
target_poly = Polygon(Border(target_path), [])

primary.setup_phases(target_velocity, True)

# on_source, on_source_layers = primary.project_transmitted_on_source()
# on_sample = primary.project_transmitted_on_sample()

on_source_alt = primary.project_on_source_alternate()
on_sample_alt = primary.project_on_sample_alternate()

# Set up the secondary spectrometer
# Only one detector per analyzer energy for now
# BIFROST analyzers are Pyrolytic Graphite and (I think) use the (002) reflection
tau_PG002 = 1.87325  # inverse angstrom
d_PG002 = 2 * pi / tau_PG002
# and a 60' mosaic?
mosaic = pi / 180
# 5-analyzers with energies between 2.5 and 5 meV (?)
ef = [2.7, 3.2, 3.8, 4.4, 5.0]
kf = [sqrt(x / 2.0722) for x in ef]
# half scattering angles. real values will also depend on:
#   which tube in a triplet the pixel is in (prismatic focusing)
#   and pixel position along each tube (scattering geometry)
ths = [arcsin(tau_PG002 / 2 / k) for k in kf]
# path lengths to each detector increase with final energy, but any numbers here are guesses
# pls = [1.8, 1.9, 2.0, 2.1, 2.2]  # m
pls = [1.1+1.189, 1.283+1.316, 1.342+1.42, 1.443+1.521, 1.544+1.623] # m

# slowest = 1 / max([x.max(1) for x in on_sample_alt])
slowest = primary.source.slowest
fastest = 1 / min([x.min(1) for x in on_sample_alt])
vs = (slowest, sqrt(2) * fastest)
detectors = [AnalyzerArm(name='detector', velocity=vs, nominal=(pl, pl), d_spacing=d_PG002, angle=th, mosaic=mosaic) for pl, th in zip(pls, ths)]
secondary = SecondarySpectrometer(detectors)

# fig, axs, handles = figure_source_sample(to_meV(on_source_alt), to_meV(on_sample_alt), ylabel=r'$E$ / meV')

after_sample = secondary.sample_scatter(on_sample_alt)
on_detector = secondary.project_on_detectors(on_sample_alt)


source_meV = to_meV(on_source_alt)
sample_meV = to_meV(on_sample_alt)
after_meV = to_meV(after_sample)
detector_meV = [to_meV(x) for x in on_detector]

# figure_source_sample_detector(on_source_alt, on_sample_alt, [after_sample], plottype=None, repeat=1/ess_frequency)
figure_source_sample_detector(source_meV, sample_meV, detector_meV, plottype='energy', repeat=1/primary.source.frequency)

pp.show()