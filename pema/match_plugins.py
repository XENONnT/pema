import strax
import straxen
import numba
from immutabledict import immutabledict
import numpy as np
import pema
import logging
export, __all__ = strax.exporter()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('Pema matching')


@export
class MatchPeaks(strax.Plugin):
    __version__ = '0.0.5'
    depends_on = ('truth', 'peak_basics')
    provides = ('truth_matched', 'peaks_matched')
    data_kind = immutabledict(truth_matched='truth',
                              peaks_matched='peaks')

    def infer_dtype(self):
        dtypes = {}
        for dtype_for in ('truth', 'peaks'):
            match_to = 'peaks' if dtype_for == 'truth' else 'truth'
            dtype = strax.dtypes.time_fields + [
                ((f'Id of element in {dtype_for}', 'id'), np.int64),
                ((f'Outcome of matching to {match_to}', 'outcome'), pema.matching.OUTCOME_DTYPE),
                ((f'Id of matching element in {match_to}', 'matched_to'), np.int64)
            ]
            dtypes[dtype_for + '_matched'] = dtype
        return dtypes

    def compute(self, truth, peaks):
        log.debug(f'Starting {self.__class__.__name__}')

        log.debug(f'Sort by time and add area')
        # Shouldn't be needed, just double checking
        truth = truth.copy()
        truth.sort(order='time')

        truth = pema.append_fields(truth, 'area', truth['n_photon'])
        # hack endtime
        log.warning(f'Patching endtime in the truth')
        truth['endtime'] = truth['t_last_photon'].copy()

        log.info('Starting matching')
        truth_vs_peak, peak_vs_truth = pema.match_peaks(truth, peaks)

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


@export
class TruthExtended(strax.MergeOnlyPlugin):
    __version__ = '0.0.0'
    depends_on = ('truth', 'truth_matched')
    provides = 'truth_extended'
    data_kind = 'truth'
    save_when = strax.SaveWhen.TARGET


@export
class PeaksExtended(strax.MergeOnlyPlugin):
    __version__ = '0.0.0'
    depends_on = ('peak_basics', 'peaks_matched')
    provides = 'peaks_extended'
    data_kind = 'peaks'
    save_when = strax.SaveWhen.TARGET


@export
@strax.takes_config(
    strax.Option('dpe_fraction', default=0.219,
                 help='Probability of double photon emission (conversion '
                      'between n_photon in truth and area in peaks)'),
    strax.Option('penalty_s2_by',
                 default=(('misid_as_s1', -1.),
                          ('split_and_misid', -1.),
                          ),
                 help='Add a penalty to the acceptance fraction if the peak '
                      'has the outcome. Should be tuple of tuples where each '
                      'tuple should have the format of '
                      '(outcome, penalty_factor)'),
    strax.Option('min_s2_bias_rec', default=0.65,
                 help='If the '),
)
class AcceptanceComputer(strax.Plugin):
    __version__ = '0.0.0'
    depends_on = ('truth_extended', 'peaks_extended')
    provides = 'match_acceptance'
    data_kind = 'truth'

    dtype = strax.dtypes.time_fields + [
        ((f'Is the peak tagged "found" in the reconstructed data',
          'is_found'),
         np.bool_),
        ((f'Acceptance of the peak can be negative for penalized reconstruction',
          'acceptance_fraction'),
         np.float64),
        ((f'Reconstruction bias 1 is perfect, 0.1 means incorrect',
          'rec_bias'),
         np.float64),
    ]

    def compute(self, truth, peaks):
        res = np.zeros(len(truth), self.dtype)

        res['time'] = truth['time']
        res['endtime'] = strax.endtime(truth)
        res['is_found'] = truth['outcome'] == 'found'

        rec_bias = np.zeros(len(truth), dtype=np.float64)
        rec_bias = self.compute_rec_bias(truth, peaks, rec_bias)
        rec_bias /= (1 + self.config['dpe_fraction'])
        res['rec_bias'] = rec_bias

        s1_mask = truth['type'] == 1
        res['acceptance_fraction'][s1_mask] = res['is_found'][s1_mask].astype(np.float)

        s2_mask = truth['type'] == 2
        s2_outcomes = truth['outcome'][s2_mask].copy()
        s2_acceptance = (rec_bias[s2_mask] > self.config['min_s2_bias_rec']).astype(np.float)
        for outcome, penalty in self.config['penalty_s2_by']:
            s2_out_mask = s2_outcomes == outcome
            s2_acceptance[s2_out_mask] = penalty
        res['acceptance_fraction'][s2_mask] = s2_acceptance
        return res

    @staticmethod
    @numba.njit
    def compute_rec_bias(truth, peaks, buffer, no_peak_found=pema.matching.INT_NAN):
        for ti, t in enumerate(truth):
            peak_i = t['matched_to']
            if peak_i != no_peak_found:
                if t['type'] == peaks[peak_i]['type']:
                    frac = peaks[peak_i]['area'] / t['n_photon']
                    buffer[ti] = frac
                    continue
            buffer[ti] = 0
        return buffer


class AcceptanceExtended(strax.MergeOnlyPlugin):
    __version__ = '0.0.0'
    depends_on = ('match_acceptance', 'truth_extended')
    provides = 'match_acceptance_extended'
    data_kind = 'truth'
    save_when = strax.SaveWhen.TARGET

