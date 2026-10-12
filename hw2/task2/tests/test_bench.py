import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from bench import Bench, Settings


def settings(**kwargs):
    return Settings(**dict(dict(macropixel_size=2,focus_size=1,background_radius=10,
                               background_regions=4,iterations=300),**kwargs))


def test_macro_aggregation_and_noise():
    bench=Bench(settings())
    bits=np.arange(64)%2
    expanded=np.repeat(np.repeat(bits.reshape(8,8),2,axis=0),2,axis=1)
    assert np.allclose(bench.A@bits,bench.T@expanded.ravel())
    field=bench.A@bits
    assert not np.array_equal(bench.shot(field),bench.shot(field))
    assert not any((m & bench.focus).any() for m in bench.background)


def test_search_improves_and_is_reproducible(tmp_path):
    a=Bench(settings()).run(tmp_path/'a')
    b=Bench(settings()).run(tmp_path/'b')
    assert a['enhancement'] > a['initial_enhancement']*3
    assert a['enhancement']==b['enhancement']
    assert (tmp_path/'a'/'camera.png').is_file()


def test_reject_invalid_geometry():
    with pytest.raises(ValueError): settings(macropixel_size=3)
    with pytest.raises(ValueError): settings(focus_size=2)
    with pytest.raises(ValueError): settings(macropixel_size=8,flips=5)
