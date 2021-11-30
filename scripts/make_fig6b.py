"""
Make the plot of frameshifting rates in Fig 6b.
"""

from matplotlib import pyplot as plt
import numpy as np
np.random.seed(1)


prf_rates = {
        "92 nt": np.asarray([17.10214, 16.90036, 17.69119]),
        "2924 nt": np.asarray([33.69146, 39.72798, 54.35715]),
}

means = {construct: np.mean(prfs) for construct, prfs in prf_rates.items()}
bias_correction_factor = 0.8862269255
stdevs = {construct: np.sqrt(
        np.sum((prf_rates[construct] - means[construct])**2) / (len(prf_rates[construct]) - 1)
    ) / bias_correction_factor for construct, prfs in prf_rates.items()}
sems = {construct: stdevs[construct] / np.sqrt(len(prf_rates[construct])) for construct in prf_rates}

x = np.asarray([1, 1, 1, 2, 2, 2])
jitter = np.random.randn(len(x)) * 0.03
y = np.hstack([prf_rates["92 nt"], prf_rates["2924 nt"]])
plt.scatter(x + jitter, y)
x = np.asarray([1, 2])
y = np.hstack([means["92 nt"], means["2924 nt"]])
yerr = np.hstack([sems["92 nt"], sems["2924 nt"]])
plt.errorbar(x, y, yerr)
plt.ylim((0, 60))
plt.savefig("fig6b.pdf")
plt.close()

