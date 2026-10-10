from pathlib import Path
import numpy as np
import scipy.io

FS = 610.35
DT = 1.0 / FS
START, N, N_TRAIN = 42650, 4096, 3072
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "SNLS80mV.mat"


def load_excerpt():
    """Return centred input u and output y of the 4096-sample excerpt."""
    mat = scipy.io.loadmat(DATA_PATH)
    u = mat["V1"].ravel()[START:START + N]
    y = mat["V2"].ravel()[START:START + N]
    assert len(u) == len(y) == N
    return u - u.mean(), y - y.mean()