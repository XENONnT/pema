from unittest import TestCase
import numpy as np
import pema

def test_import():

    from pema import matching
    from pema import MatchPeaks
    from pema import append_fields
    try:
        from pema import _plot_peak_matching_histogram
        raise ValueError(
            '_plot_peak_matching_histogram should not be available at top level')
    except (ImportError, ValueError, FileNotFoundError, ModuleNotFoundError):
        # Good we cannot import this (which is what we want)
        pass
    print('done')


class SimpleTests(TestCase):
    """Odd bunch of tests that can be removed if needed"""
    def check_raises_check_args(self):
        truth = np.zeros(1, dtype=[(('bla bla', 'bla'), np.int8)])
        with self.assertRaises(ValueError):
            pema.compare_plots._check_args(truth, run_id=None)
        with self.assertRaises(ValueError):
            pema.compare_plots._check_args([], truth_vs_custom=truth, run_id='bla')

    def test_show(self):
        pema.compare_plots('a', 'b', show=True)

    def test_warnings_for_matching(self):
        with self.assertRaises(ValueError):
            peak1 = np.zeros(1, dtype=[(('bla bla', 'bla'), np.int8)])
            peak2 = peak1
            pema.match_peaks(peak1, peak2)
