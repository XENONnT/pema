import strax
import straxen
from immutabledict import immutabledict
import numpy as np
import peak_matching


class SortTruth(strax.Plugin):
    """Temporary patch if one has"""
    depends_on = 'truth'
    provides = 'truth_sorted'

    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['truth'].dtype_for('truth'))

    def compute(self, truth):
        res = truth.copy()
        res.sort(order='time')
        return res


class MatchPeaks(strax.Plugin):
    __version__ = '0.0.1'
    depends_on = ('truth_sorted', 'peak_basics')
    provides = ('truth_matched', 'peaks_matched')
    data_kind = immutabledict(truth_matched='truth',
                              peaks_matched='peaks')

    def infer_dtype(self):
        dtypes = {}
        for dtype_for in ('truth', 'peaks'):
            match_to = 'peaks' if dtype_for == 'truth' else 'truth'
            dtype = strax.dtypes.time_fields + [
                ((f'Id of element in {dtype_for}', 'id'), np.float64),
                ((f'Outcome of matching to {match_to}', 'outcome'), peak_matching.OUTCOME_DTYPE),
                ((f'Id of matching element in {match_to}', 'matched_to'), np.float64)
            ]
            dtypes[dtype_for + '_matched'] = dtype
        return dtypes

    def compute(self, truth, peaks):
        truth = peak_matching.append_fields(truth, 'area', truth['n_photon'])
        truth_vs_peak, peak_vs_truth = peak_matching.match_peaks_strax(truth, peaks)

        # Truth
        res_truth = {}
        for k in self.dtype['truth_matched'].names:
            res_truth[k] = truth_vs_peak[k]

        # Peaks
        res_peak = {}
        for k in self.dtype['peaks_matched'].names:
            res_peak[k] = peak_vs_truth[k]

        return {'truth_matched': res_truth,
                'peaks_matched': res_peak}
