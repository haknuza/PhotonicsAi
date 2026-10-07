import numpy as np
from scipy.signal import find_peaks, peak_widths
from fabry_perot import fabry_perot

_STATE = {}   # stores the last spectrum: wavelengths, transmission, step

def run_spectrum(length_um: float, R: float, lam_min_nm: float, lam_max_nm: float) -> str:
    """Simulate a Fabry-Perot cavity and store the spectrum.

    Args:
        length_um: cavity length in micrometers
        R: mirror reflectivity, between 0.05 and 0.99
        lam_min_nm: start wavelength in nm
        lam_max_nm: end wavelength in nm
    """
    # 1. validate inputs -> return an error STRING if bad (don't raise)
    if not (0.05 <= R <= 0.99):
        return f"Error: Reflectivity R must be between 0.05 and 0.99. got R={R}"
    if lam_min_nm >= lam_max_nm:
        return f"Error: lam_min_nm must be less than lam_max_nm. got lam_min_nm={lam_min_nm}, lam_max_nm={lam_max_nm}"
    if length_um <= 0:
        return f"Error: length_um must be positive. got length_um={length_um}"
    
    # 2. choose step (see below)
    lam_c = (lam_min_nm + lam_max_nm) / 2
    fsr = lam_c**2 / (2 * length_um * 1e3)  # in nm 
    fwhm = fsr * (1 - R) / (np.pi * np.sqrt(R))  # in nm
    step = fwhm / 10  # choose step as 1/10 of the FWHM to resolve the peaks
    finesse = np.pi * np.sqrt(R) / (1 - R)
    if (lam_max_nm - lam_min_nm) / step < 50:
        return f"Error: Increase the wavelength range. Computed number of points={(lam_max_nm - lam_min_nm) / step}" 
    
    # 3. wavelengths = np.arange(...), T = fabry_perot(...)
    if step <= 0:
        return f"Error: Computed step size is non-positive. Check your parameters. Computed step={step}"
    if (lam_max_nm - lam_min_nm) / step > 10000:
        return f"Error: Reduce the wavelength range or increase the step size. Computed number of points={(lam_max_nm - lam_min_nm) / step}"
    wavelengths = np.arange(lam_min_nm, lam_max_nm, step)
    
    # 4. _STATE.update(lam=..., T=..., step=...)
    _STATE.update(lam=wavelengths, T=fabry_perot(wavelengths, length_um, R), step=step)
    
    # 5. return a short summary: number of points, step, max/min T
    return f"Spectrum computed with {len(wavelengths)} points, step={step:.3f} nm, T_max={np.max(_STATE['T']):.3f}, T_min={np.min(_STATE['T']):.3f}"

def find_resonances() -> str:
    """Find the transmission peaks of the last spectrum. Returns their wavelengths in nm.
    Takes no arguments, gets the last spectrum from _STATE in run_spectrum.returns a string with the peak wavelengths."""
    if not _STATE:
        return "Error: No spectrum computed yet. Please run run_spectrum() first."
    
    # 1. use scipy.signal.find_peaks to find peaks in _STATE['T']
    peaks, _ = find_peaks(_STATE['T'], height=0.5)  # only consider peaks with height > 0.5
    
    # 2. return a string with the peak wavelengths
    peak_wavelengths = _STATE['lam'][peaks]
    return f"Found {len(peak_wavelengths)} resonances at wavelengths (nm): {', '.join(f'{w:.2f}' for w in peak_wavelengths)}"

def measure_finesse() -> str:
    """Measure free spectral range, FWHM and finesse of the last spectrum.
    Takes no arguments, gets the last spectrum from _STATE in run_spectrum. Returns a string with the finesse."""
    if not _STATE:
        return "Error: No spectrum computed yet. Please run run_spectrum() first."
    
    # 1. use scipy.signal.peak_widths to find FWHM of peaks
    peaks, _ = find_peaks(_STATE['T'], height=0.5)
    
    if len(peaks) < 2:
        return "Error: Not enough peaks found to measure finesse. Please ensure the spectrum has multiple resonances. increase the wavelength range or adjust the cavity length and reflectivity."
    
    
    fsr = np.mean(np.diff(_STATE['lam'][peaks]))  # average spacing between peaks
    widths = peak_widths(_STATE['T'], peaks, rel_height=0.5 )[0] * _STATE['step']  # convert widths from indices to nm
    finesse = fsr / np.mean(widths)
    
    # 2. return a string with the finesse
    return f"Measured finesse: FSR={fsr:.3f} nm, FWHM={np.mean(widths):.3f} nm, Finesse={finesse:.3f}"
