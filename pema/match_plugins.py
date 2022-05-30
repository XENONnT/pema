import warnings

import strax
import numba
import numpy as np
import pema
import logging
import straxen
from .matching import INT_NAN

export, __all__ = strax.exporter()

log = logging.getLogger('Pema matching')


@export
class MatchPeaks(strax.OverlapWindowPlugin):
    """
    Match WFSim truth to the outcome peaks. To this end use the
        matching algorithm of pema. Assign a peak-id to both the truth
        and the reconstructed peaks to be able to match the two. Also
        define the outcome of the matching (see pema.matching for
        possible outcomes).
    """
    __version__ = '0.5.0'
    depends_on = ('truth', 'truth_id', 'peak_basics', 'peak_id')
    provides = 'truth_matched'
    data_kind = 'truth'

    truth_lookup_window = straxen.URLConfig(
        default=int(1e9),
        help='Look back and forth this many ns in the truth info',
    )

    def compute(self, truth, peaks):
        log.debug(f'Starting {self.__class__.__name__}')
        truth = pema.append_fields(truth, 'area', truth['raw_area'])

        # hack endtime
        log.warning(f'Patching endtime in the truth')
        truth['endtime'] = truth['t_last_photon'].copy()

        log.info('Starting matching')
        truth_vs_peak, peak_vs_truth = pema.match_peaks(truth, peaks)

        # copy to the result buffer
        res_truth = np.zeros(len(truth), dtype=self.dtype)
        for k in self.dtype.names:
            res_truth[k] = truth_vs_peak[k]

        return res_truth

    def get_window_size(self):
        return self.config['truth_lookup_window']

    def infer_dtype(self):
        dtype = strax.dtypes.time_fields + [
            ((f'Id of element in truth', 'id'), np.int64),
            ((f'Outcome of matching to peaks', 'outcome'), pema.matching.OUTCOME_DTYPE),
            ((f'Id of matching element in peaks', 'matched_to'), np.int64)
        ]
        return dtype


@export
class AcceptanceComputer(strax.Plugin):
    """
    Compute the acceptance of the matched peaks. This is done on the
    basis of arbitrary settings to allow better to disentangle
    possible scenarios that might be undesirable (like splitting
    an S2 into small S1 signals that could affect event
    reconstruction).
    """
    __version__ = '2.0.2'
    depends_on = ('truth', 'truth_matched', 'peak_basics', 'peak_id')
    provides = 'match_acceptance'
    data_kind = 'truth'

    keep_peak_fields = straxen.URLConfig(
        default=('area', 'range_50p_area', 'area_fraction_top', 'rise_time', 'tight_coincidence'),
        help='Add the reconstructed value of these variables',
    )
    penalty_s2_by = straxen.URLConfig(
        default=(('misid_as_s1', -1.), ('split_and_misid', -1.),),
        help='Add a penalty to the acceptance fraction if the peak has the '
             'outcome. Should be tuple of tuples where each tuple should '
             'have the format of (outcome, penalty_factor)',
    )
    min_s2_bias_rec = straxen.URLConfig(
        default=0.85,
        help='If the S2 fraction is greater or equal than this, consider a '
             'peak successfully found even if it is split or chopped.',
    )

    def compute(self, truth, peaks):
        res = np.zeros(len(truth), self.dtype)
        res['time'] = truth['time']
        res['endtime'] = strax.endtime(truth)
        res['is_found'] = truth['outcome'] == 'found'

        peak_idx = truth['matched_to']
        mask = peak_idx != INT_NAN
        if np.sum(mask):
            # need to get at least one peak for each, even if we are going to remove those later
            sel_from_peaks = peak_idx[mask]
            sel_peaks = peaks[get_idx(sel_from_peaks, peaks['id'], INT_NAN)]

            if len(sel_peaks) != len(sel_from_peaks):
                raise ValueError(f'Got {len(sel_peaks)} != {len(sel_from_peaks)}')
            not_match = sel_from_peaks != sel_peaks['id']
            if np.any(not_match):
                for i, t_i, p_i in zip(not_match, sel_from_peaks, sel_peaks['id']):
                    print(i, t_i, p_i)
                raise ValueError
            for k in self.keep_peak_fields:
                res[f'rec_{k}'][mask] = sel_peaks[k]

        res['rec_bias'] = res['rec_area'] / truth['raw_area']

        # S1 acceptance is simply is the peak found or not
        s1_mask = truth['type'] == 1
        res['acceptance_fraction'][s1_mask] = res['is_found'][s1_mask].astype(np.float64)

        # For the S2 acceptance we calculate an arbitrary acceptance
        # that takes into account penalty factors and that S2s may be
        # split (as long as their bias fraction is not too small).
        s2_mask = truth['type'] == 2
        s2_outcomes = truth['outcome'][s2_mask].copy()
        s2_acceptance = (res[s2_mask]['rec_bias'] > self.config['min_s2_bias_rec']).astype(np.float64)
        for outcome, penalty in self.config['penalty_s2_by']:
            s2_out_mask = s2_outcomes == outcome
            s2_acceptance[s2_out_mask] = penalty

        # now update the acceptance fraction in the results
        res['acceptance_fraction'][s2_mask] = s2_acceptance

        return res

    def infer_dtype(self):
        dtype = strax.dtypes.time_fields + [
            ((f'Is the peak tagged "found" in the reconstructed data',
              'is_found'), np.bool_),
            ((f'Acceptance of the peak can be negative for penalized reconstruction',
              'acceptance_fraction'),
             np.float64),
            ((f'Reconstruction bias 1 is perfect, 0.1 means incorrect',
              'rec_bias'),
             np.float64),
        ]
        for descr in self.deps['peak_basics'].dtype_for('peak_basics').descr:
            # Add peak fields
            field = descr[0][1]
            if field in self.keep_peak_fields:
                dtype += [((descr[0][0], f'rec_{field}'), descr[1])]
        return dtype

    def setup(self):
        assert 'area' in self.keep_peak_fields


class AcceptanceExtended(strax.MergeOnlyPlugin):
    """Merge the matched acceptance to the extended truth"""
    __version__ = '0.1.0'
    depends_on = ('match_acceptance', 'truth', 'truth_id', 'truth_matched')
    provides = 'match_acceptance_extended'
    data_kind = 'truth'
    save_when = strax.SaveWhen.TARGET

    def setup(self):
        warnings.warn(f'match_acceptance_extended is deprecated use truth_extended',
                      DeprecationWarning)
        super().setup()


@export
class TruthExtended(strax.MergeOnlyPlugin):
    """Merge the matched acceptance to the extended truth"""
    __version__ = '0.1.0'
    depends_on = ('match_acceptance', 'truth', 'truth_id', 'truth_matched')
    provides = 'truth_extended'
    data_kind = 'truth'
    save_when = strax.SaveWhen.TARGET


@strax.takes_config(
    strax.Option('truth_lookup_window',
                 default=int(1e9),
                 help='Look back and forth this many ns in the truth info'),
)
class MatchEvents(strax.OverlapWindowPlugin):
    """
    Match WFSim truth to the outcome peaks. To this end use the
        matching algorithm of pema. Assign a peak-id to both the truth
        and the reconstructed peaks to be able to match the two. Also
        define the outcome of the matching (see pema.matching for
        possible outcomes).
    """
    __version__ = '0.1.0'
    depends_on = ('truth', 'events')
    provides = 'truth_events'
    data_kind = 'truth_events'

    dtype = strax.dtypes.time_fields + [
        ((f'First event number in event datatype within the truth event', 'start_match'), np.int64),
        ((f'Last (inclusive!) event number in event datatype within the truth event', 'end_match'),
         np.int64),
        ((f'Outcome of matching to events', 'outcome'), pema.matching.OUTCOME_DTYPE),
        ((f'Truth event number', 'truth_number'), np.int64),
    ]

    def compute(self, truth, events):
        unique_numbers = np.unique(truth['event_number'])
        res = np.zeros(len(unique_numbers), self.dtype)
        res['truth_number'] = unique_numbers
        fill_start_end(truth, res)
        assert np.all(res['endtime'] > res['time'])
        assert np.all(np.diff(res['time']) > 0)

        tw = strax.touching_windows(events, res)
        tw_start = tw[:, 0]
        tw_end = tw[:, 1] - 1
        found = tw_end - tw_start > 0
        diff = np.diff(tw, axis=1)[:, 0]

        res['start_match'][found] = events[tw_start[found]]['event_number']
        res['end_match'][found] = events[tw_end[found]]['event_number']
        res['outcome'] = self.outcomes(diff)
        res['start_match'][~found] = pema.matching.INT_NAN
        res['end_match'][~found] = pema.matching.INT_NAN
        return res

    def get_window_size(self):
        return self.config['truth_lookup_window']

    @staticmethod
    def outcomes(diff):
        outcome = np.empty(len(diff), dtype=pema.matching.OUTCOME_DTYPE)
        not_found_mask = diff < 0
        one_found_mask = diff == 0
        many_found_mask = diff >= 1
        outcome[not_found_mask] = 'missed'
        outcome[one_found_mask] = 'found'
        outcome[many_found_mask] = 'split'
        return outcome


class PeakId(strax.Plugin):
    """Add id field to datakind"""
    depends_on = 'peak_basics'
    provides = 'peak_id'
    data_kind = 'peaks'
    __version__ = '0.0.0'

    peaks_seen = 0
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = strax.time_fields
        id_field = [((f'Id of element in {self.data_kind}', 'id'), np.int64), ]
        return dtype + id_field

    def compute(self, peaks):
        res = np.zeros(len(peaks), dtype=self.dtype)
        res['time'] = peaks['time']
        res['endtime'] = peaks['endtime']
        peak_id = np.arange(len(peaks)) + self.peaks_seen
        res['id'] = peak_id
        self.peaks_seen += len(peaks)
        return res


class TruthId(PeakId):
    depends_on = 'truth'
    provides = 'truth_id'
    data_kind = 'truth'
    __version__ = '0.0.0'

    def compute(self, truth):
        assert_ordered_truth(truth)
        return super().compute(truth)


@numba.njit()
def fill_start_end(truth, truth_event):
    for i, ev_i in enumerate(truth_event['truth_number']):
        mask = truth['event_number'] == ev_i
        start = truth[mask]['time'].min()
        stop = truth[mask]['endtime'].max()
        truth_event['time'][i] = start
        truth_event['endtime'][i] = stop


def assert_ordered_truth(truth):
    assert np.all(np.diff(truth['time']) >= 0), "truth is not sorted!"


@numba.njit
def get_idx(search_item, in_list, not_found=-99999):
    """Get index in <in_list> where the value is <searc_value>

    Assumes that <in_list> is sorted!

    :returns:
        a list of length (search item) where each value refers to the item in <in_list>
    """
    result = np.ones(len(search_item), dtype=np.int64) * not_found
    look_from = 0
    for i, search in enumerate(search_item):
        for k, v in enumerate(in_list[look_from:]):
            if v == search:
                result[i] = look_from + k
                look_from += k
                break
    for r in result:
        if r == not_found:
            raise ValueError()
    return result
