from os import name

import matplotlib.pyplot as plt
import numpy as np

def fabry_perot(wavelengths_nm, length_um, R, n=1.0, theta=0.0):
    """Transmission of a lossless Fabry-Perot cavity (Airy function)."""
    lam = np.asarray(wavelengths_nm) * 1e-9
    L = length_um * 1e-6
    delta = 4 * np.pi * n * L * np.cos(theta) / lam
    F = 4 * R / (1 - R) ** 2
    return 1.0 / (1.0 + F * np.sin(delta / 2) ** 2)


wavelengths = np.linspace(400, 700, 1000)  # Wavelengths from 400 nm to 700 nm
length = 10.0  # Length of the cavity in micrometers
R = 0.9  # Reflectivity of the mirrors
transmission = fabry_perot(wavelengths, length, R)
if name == "__main__":
    plt.plot(wavelengths, transmission)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("Fabry-Perot Cavity Transmission")
    plt.show()
