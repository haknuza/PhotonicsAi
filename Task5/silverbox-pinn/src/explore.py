import numpy as np
import matplotlib.pyplot as plt

from data import DT, N, N_TRAIN, START, load_excerpt

u, y = load_excerpt()
t = np.arange(N) * DT            # time axis in seconds

# ---------- 1. sanity checks ----------
print("samples:", len(u), len(y))                              # expect 4096 4096
print("peak |y| [V]:", np.max(np.abs(y)))                      # expect near 0.2
print("test RMS of y [V]:", np.sqrt(np.mean(y[N_TRAIN:] ** 2)))  # expect near 0.06

freqs = np.fft.rfftfreq(N, d=DT)
mag_u = np.abs(np.fft.rfft(u)) / N
mag_y = np.abs(np.fft.rfft(y)) / N
print("dominant frequency of y [Hz]:", freqs[np.argmax(mag_y[1:]) + 1])  # expect near 70

# ---------- 2. zoom on the first 150 samples ----------
K = 150
fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1[0].plot(t[:K], u[:K], marker=".")
ax1[0].set_ylabel("u [V]")
ax1[1].plot(t[:K], y[:K], marker=".")
ax1[1].set_ylabel("y [V]")
ax1[1].set_xlabel("Time [s]")
for a in ax1:
    a.grid(True, linestyle="--", alpha=0.6)
fig1.suptitle("Zoom: first 150 samples (smooth means no big measurement noise)")
fig1.tight_layout()

# ---------- 3. spectra of u and y ----------
fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax2[0].semilogy(freqs, mag_u)
ax2[0].set_ylabel("|U| [V]")
ax2[1].semilogy(freqs, mag_y)
ax2[1].set_ylabel("|Y| [V]")
ax2[1].set_xlabel("Frequency [Hz]")
for a in ax2:
    a.grid(True, which="both", linestyle="--", alpha=0.6)
fig2.suptitle(f"Spectra of u and y, samples {START}:{START + N}")
fig2.tight_layout()

# ---------- 4. acceleration against y ----------
v = np.gradient(y, DT)           # velocity from finite differences
acc = np.gradient(v, DT)         # acceleration from finite differences
s = slice(2, -2)                 # drop the less accurate edge samples

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.scatter(y[s], acc[s], s=3, alpha=0.4)
ax3.set_xlabel("y [V]")
ax3.set_ylabel("acceleration [V/s²]")
ax3.set_title("Acceleration against y (a cloud is normal: v and u also act)")
ax3.grid(True, linestyle="--", alpha=0.6)
fig3.tight_layout()

print("acceleration, typical size [V/s²]:", np.std(acc[s]))

plt.show()    # show all three figures at once