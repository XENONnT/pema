import strax
import straxen
import numpy as np
import matplotlib.pyplot as plt
from straxen.analyses.waveform_plot import time_and_samples, seconds_range_xaxis
import pema
from tqdm.notebook import tqdm


@straxen.mini_analysis(
    requires=('truth'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_instructions(
        truth,
        seconds_range,
):
    for pi, peak in enumerate(truth):
        hatch_cycle = ['/', '*', '+', '|'] * 20
        print(peak, peak.dtype)
        _t_range = peak[['time', 'endtime']]
        x = np.array(list(_t_range))
        y = peak['amp'] / np.diff(x)

        stype = peak['type']
        plt.gca()
        plt.fill_between(x / 1e9, 0, y,
                         color={1: 'blue',
                                2: 'green',
                                0: 'gray',
                                6: 'orange',
                                }[stype],
                         label=f'Peak S{stype}',
                         alpha=0.4,
                         hatch=hatch_cycle[pi]
                         )


@straxen.mini_analysis(
    requires=('peaks', 'peak_basics'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_peaks(peaks,
               seconds_range,
               t_reference,
               include_info=None,
               show_largest=100,
               single_figure=True,
               figsize=(10, 4),
               xaxis=True):
    if single_figure:
        plt.figure(figsize=figsize)
    plt.axhline(0, c='k', alpha=0.2)

    peaks = peaks[np.argsort(-peaks['area'])[:show_largest]]
    peaks = strax.sort_by_time(peaks)

    for p in peaks:
        plot_peak(p,
                  t0=t_reference,
                  include_info=include_info,
                  color={0: 'gray', 1: 'b', 2: 'g'}[p['type']])

    if xaxis == 'since_start':
        seconds_range_xaxis(seconds_range, t0=seconds_range[0])
    elif xaxis:
        seconds_range_xaxis(seconds_range)
        plt.xlim(*seconds_range)
    else:
        plt.xticks([])
        plt.xlim(*seconds_range)
    plt.ylabel("Intensity [PE/ns]")
    if single_figure:
        plt.tight_layout()


def plot_peak(p, t0=None, center_time=True, include_info=None, **kwargs):
    x, y = time_and_samples(p, t0=t0)
    kwargs.setdefault('linewidth', 1)

    # Plot waveform
    plt.plot(x, y,
             drawstyle='steps-pre',
             **kwargs)
    if 'linewidth' in kwargs:
        del kwargs['linewidth']
    kwargs['alpha'] = kwargs.get('alpha', 1) * 0.2
    plt.fill_between(x, 0, y, step='pre', linewidth=0, **kwargs)

    # Mark extent with thin black line
    plt.plot([x[0], x[-1]], [y.max(), y.max()],
             c='k', alpha=0.3, linewidth=1)

    # Mark center time with thin black line
    if center_time:
        if t0 is None:
            t0 = p['time']
        ct = (p['center_time'] - t0) / int(1e9)
        plt.axvline(ct, c='k', alpha=0.4, linewidth=1, linestyle='--')
    if include_info:
        info_str = '\n'.join([f'{inf}: {p[inf]:.1f}' for inf in include_info])
        plt.text(x[-1],
                 y.max(),
                 info_str,
                 fontsize='xx-small',
                 ha='left',
                 va='top',
                 alpha=0.8,
                 bbox=dict(boxstyle="round", fc="w", alpha=0.5)
                 )


def compare_outcomes(st_default, truth_vs_default,
                     st_custom, truth_vs_custom,
                     fuzz=500,
                     plot_fuzz=1000,
                     max_peaks=10,
                     default_label='default',
                     custom_label='custom',
                     fig_dir=None,
                     show=True,
                     randomize=True,
                     only_different=True
                     ):
    if only_different:
        peaks_idx = np.where(truth_vs_default['outcome'] != truth_vs_custom['outcome'])[0]
    else:
        peaks_idx = np.arange(len(truth_vs_default))
    if randomize:
        np.random.shuffle(peaks_idx)
    for peak_i in tqdm(peaks_idx[:max_peaks]):
        try:
            t_range = (truth_vs_custom[peak_i]['time'] - fuzz,
                       truth_vs_custom[peak_i]['endtime'] + fuzz)

            f, axes = plt.subplots(3, 1,
                                   figsize=(10, 10),
                                   gridspec_kw={'height_ratios': [0.5, 1, 1]})
            xlim = (t_range[0] - plot_fuzz) / 1e9, (t_range[1] + plot_fuzz) / 1e9

            plt.sca(axes[0])
            plt.title('Instructions')
            start_end = np.zeros(1, dtype=strax.time_fields)
            start_end['time'] = t_range[0]
            start_end['endtime'] = t_range[1]
            run_mask = truth_vs_custom['run_id'] == truth_vs_custom[peak_i]['run_id']
            for pk, pi in enumerate(
                    range(*strax.touching_windows(truth_vs_custom[run_mask], start_end)[0])):
                tpeak = truth_vs_custom[run_mask][pi]
                hatch_cycle = ['/', '*', '+', '|']
                _t_range = tpeak[['time', 'endtime']]
                x = np.array(list(_t_range))
                y = tpeak['n_photon'] / np.diff(x)
                ct = tpeak['t_mean_photon']
                stype = tpeak['type']
                plt.gca()
                plt.fill_between([x[0] / 1e9, ct / 1e9, x[-1] / 1e9, ],
                                 [0, 0, 0], [0, 2 * y[0], 0],
                                 color={1: 'blue',
                                        2: 'green',
                                        0: 'gray',
                                        6: 'orange',
                                        4: 'purple',
                                        }[stype],
                                 label=f'Peak S{stype}. {tpeak["n_photon"]} PE',
                                 alpha=0.4,
                                 hatch=hatch_cycle[pk]
                                 )
                plt.ylabel('Intensity [PE/ns]')
            for t in t_range:
                axvline(t / 1e9, label=f't = {t}')

            plt.legend(loc='lower left', fontsize='x-small')

            plt.xlim(*xlim)

            plt.sca(axes[1])
            plt.title(default_label)
            st_default.plot_peaks(truth_vs_custom[peak_i]['run_id'],
                                  single_figure=False,
                                  include_info=['area', 'rise_time', 'tight_coincidence'],
                                  time_range=t_range)
            for t in t_range:
                axvline(t / 1e9, label=t)
            plt.xlim(*xlim)
            plt.gca().set_xticklabels([])
            plt.xlabel('')
            plt.text(0.05, 0.95,
                     truth_vs_default[peak_i]['outcome'],
                     transform=plt.gca().transAxes,
                     ha='left',
                     va='top',
                     bbox=dict(boxstyle="round", fc="w")
                     )

            plt.text(0.05, 0.1,
                     '\n'.join(f'{prop[:10]}: {truth_vs_default[peak_i][prop]:.1f}' for prop in ['rec_bias', 'acceptance_fraction']),
                     transform=plt.gca().transAxes,
                     fontsize='small',
                     ha='left',
                     va='bottom',
                     bbox=dict(boxstyle="round", fc="w"),
                     alpha=0.8,
                     )

            plt.sca(axes[2])
            plt.title(custom_label)
            st_custom.plot_peaks(truth_vs_custom[peak_i]['run_id'],
                                 single_figure=False,
                                 include_info=['area', 'rise_time', 'tight_coincidence'],
                                 time_range=t_range)
            plt.text(0.05, 0.95,
                     truth_vs_custom[peak_i]['outcome'],
                     transform=plt.gca().transAxes,
                     ha='left',
                     va='top',
                     bbox=dict(boxstyle="round", fc="w")
                     )
            plt.text(0.05, 0.1,
                     '\n'.join(f'{prop[:10]}: {truth_vs_custom[peak_i][prop]:.1f}' for prop in ['rec_bias', 'acceptance_fraction']),
                     transform=plt.gca().transAxes,
                     fontsize='small',
                     ha='left',
                     va='bottom',
                     bbox=dict(boxstyle="round", fc="w"),
                     alpha=0.8,
                     )
            for t in t_range:
                axvline(t / 1e9, label=t)
            plt.xlim(*xlim)
            if fig_dir:
                pema.save_canvas(f'example_wf_{peak_i}', save_dir=fig_dir)
            if show:
                plt.show()
        except (ValueError, RuntimeError) as e:
            print(f'Error making {peak_i}: {type(e)}, {e}')
            plt.show()


def axvline(v, **kwargs):
    vline_color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.axvline(v, color=vline_color, **kwargs)
