import pema
import numpy as np
from hypothesis import given, strategies, example, settings


@settings(max_examples=100, deadline=None)
@given(strategies.integers(min_value=0, max_value=100),
       strategies.integers(min_value=0, max_value=2), )
@example(length=3, dtypenum=0)
def test_combine_and_flip(length, dtypenum):
    """Check that combine and flip works correctly"""
    known_dtypes = (np.bool_, np.bool, np.int)
    dtype = known_dtypes[dtypenum]
    arr1 = np.random.randint(0, 2, length).astype(dtype)
    arr2 = np.random.randint(0, 2, length).astype(dtype)
    res = (True ^ arr1) & (True ^ arr2)
    numba_res = pema.matching._combine_and_flip(arr1, arr2)
    assert np.all(res == numba_res)


@settings(max_examples=100, deadline=None)
@given(
    # Max length of the data
    strategies.integers(min_value=0, max_value=1000),
    # Max length of the truth
    strategies.integers(min_value=0, max_value=1000),
    # Max duration in [ns]
    strategies.integers(min_value=0, max_value=10000),
    # Number of peak-types in data
    strategies.integers(min_value=0, max_value=10),
    # Number of peak-types in truth
    strategies.integers(min_value=0, max_value=10),
)
@example(data_length=0, truth_length=10, max_duration=10, n_data_types=2, n_truth_types=2)
def test_matching(data_length, truth_length, max_duration, n_data_types, n_truth_types):
    dtype = [(('Start time since unix epoch [ns]', 'time'), np.int64),
             (('Exclusive end time since unix epoch [ns]', 'endtime'), np.int64),
             (('type of p', 'type'), np.int16),
             (('area of p', 'area'), np.float64)]

    data = np.zeros(data_length, dtype=dtype)
    data['time'] = np.random.randint(0, 1e9, data_length)

    data['endtime'] = data['time']
    if max_duration:
        # randint does not work for 0, 0 interval, hence we only add it if max_duration>0
        data['endtime'] += np.random.randint(0, max_duration, data_length)
    data['type'] = np.random.randint(0, n_data_types + 1, data_length)
    data.sort(order='time')

    truth = np.zeros(truth_length, dtype=dtype)
    truth['time'] = np.random.randint(0, 1e9, truth_length)
    truth['endtime'] = truth['time']
    if max_duration:
        # randint does not work for 0, 0 interval, hence we only add it if max_duration>0
        truth['endtime'] += np.random.randint(0, max_duration, truth_length)
    truth['type'] = np.random.randint(0, n_truth_types + 1, truth_length)
    truth.sort(order='time')

    d_matched, t_matched = pema.match_peaks(data, truth)
    assert len(d_matched) == len(data)
    assert len(t_matched) == len(truth)
