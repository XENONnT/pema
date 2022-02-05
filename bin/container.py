#!/usr/bin/env python
import strax
import straxen
import pema
import pema
import os
import straxen
import wfsim
import pandas as pd
import numpy as np


def basics_plots(st, runs):
    events = st.get_array(runs, 'event_info', progress_bar = True)
    plt.hist2d(events['drift_time'],
               events['s2_range_50p_area'],
               range=[[0,3e6], [0, 20000]],
               bins = 200,
               norm=LogNorm());
    plt.xlabel('Drift time [ns]')
    plt.ylabel('S2 width')
    plt.grid()
    plt.show()

    plt.hist2d(events['r']**2,
               -events['drift_time'],
               range=[[0,6000], [-3e6, 1e6]],
               bins = 100,
               norm=LogNorm());
    plt.xlabel('$R^2$ [cm$^2$]')
    plt.ylabel('Drift time [ns]')
    plt.grid()
    plt.show()

    plt.hist2d(np.log10(events['s1_area']),
               np.log10(events['s2_area']),
               range=[[0,6], [1,7]],
               bins = 200,
               norm=LogNorm());
    plt.xlabel('log10 S1 Area')
    plt.ylabel('log10 S2 Area')
    plt.grid()
    plt.show()

    peaks =  st.get_array(runs, 'peak_basics', progress_bar = True)
    kwargs = dict(
        norm=LogNorm(), range= [[0, 70], [0, 200]], bins=[70, 100]
    )
    _compare_data(peaks, 'area', 'rise_time', **kwargs)
    plt.grid(True)
    plt.title(f'Simulated peaks')
    plt.show()


def _compare_data(data, x, y, **kwargs):
    plt.hist2d(data[x],
               data[y], **kwargs)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()

    kwargs = dict(
        norm=LogNorm(), range= [[0, 70], [0, 200]], bins=[70, 100]
    )


def compute_acceptance(data):
    """
    Simple function to calculate the mean acceptance fraction of an entire
    dataset (incl. the binom. error)
    """
    total = len(data)
    found = np.sum(data['acceptance_fraction'])
    return found/total, pema.binom_interval(found, total)

def compute_bias(data):
    """
    Simple function to calculate the mean bias of an entire dataset (incl. std)
    """
    total = len(data)
    sub_sel = data['rec_bias'] > 0
    return np.mean(data['rec_bias'][sub_sel]), np.std(data['rec_bias'][sub_sel])


def si_acceptance(default_acceptence,
                  custom_acceptence,
                  si,
                  binedges,
                  on_axis='n_photon',
                  nbins=50):
    mask = default_acceptence['type'] == si
    pema.summary_plots.acceptance_plot(
        default_acceptence[mask],
        on_axis,
        binedges,
        nbins=nbins,
        plot_label=default_label,
    )
    mask = custom_acceptence['type'] == si
    pema.summary_plots.acceptance_plot(
        custom_acceptence[mask],
        on_axis,
        binedges,
        nbins=nbins,
        plot_label=custom_label,
    )
    plt.ylabel('Arb. Acceptance')
    plt.title(f"S{si} acceptance")
    plt.legend()

def acceptance_summary(si,
                       default_acceptence,
                       custom_acceptence,
                       on_axis,
                       axis_label,
                       nbins = 100,
                       plot_range = (0, 200),
                       save_name='',
                       inculde_compare = True):
    f, axes = plt.subplots(2+int(inculde_compare), 1, figsize=(10,4*(2+int(inculde_compare))), sharex=True)
    max_photons = 35
    plt.sca(axes[0])
    sel = ((default_acceptence['type'] == si)
           & (default_acceptence[on_axis] > plot_range[0])
           & (default_acceptence[on_axis] < plot_range[1])
           )
    pema.summary_plots.plot_peak_matching_histogram(default_acceptence[sel], on_axis, bin_edges = nbins)
    plt.text(0.05,0.95,
             default_label,
             transform=plt.gca().transAxes,
             ha = 'left',
             va = 'top',
             bbox=dict(boxstyle="round", fc="w")
             )
    plt.legend(loc=(1.01,0))
    plt.xlim(*plot_range)
    if inculde_compare:
        plt.sca(axes[1])
        sel = ((custom_acceptence['type'] == si)
               & (custom_acceptence[on_axis] > plot_range[0])
               & (custom_acceptence[on_axis] < plot_range[1])
               )
        print(f'cust {np.sum(sel)}')
        pema.summary_plots.plot_peak_matching_histogram(custom_acceptence[sel], on_axis, bin_edges = nbins)
        plt.text(0.05,0.95,
                 custom_label,
                 transform=plt.gca().transAxes,
                 ha = 'left',
                 va = 'top',
                 bbox=dict(boxstyle="round", fc="w")
                 )
        plt.legend(loc=(1.01,0))
        plt.xlim(*plot_range)

    plt.sca(axes[1+int(inculde_compare)])
    mask = default_acceptence['type'] == si
    pema.summary_plots.acceptance_plot(default_acceptence[mask], on_axis, plot_range, nbins=nbins,
                                       plot_label=default_label)
    if inculde_compare:
        mask = custom_acceptence['type'] == si

        pema.summary_plots.acceptance_plot(custom_acceptence[mask], on_axis, plot_range, nbins=nbins,
                                           plot_label=custom_label)
    plt.legend(loc=(1.01,0))
    plt.ylabel('Arb. acceptance faction')
    plt.xlim(*plot_range)
    plt.xlabel(axis_label)
    plt.ylim(0,1)

    plt.subplots_adjust(hspace=0)
    plt.suptitle(f'S{si} Acceptance', y=0.9)
    pema.save_canvas(f'{si}_acceptance_detailed_{save_name}', save_dir=fig_dir)

def remake(sts, runs, ts, progress=False):
    for _st in sts:
        for r in runs:
            for t in ts:
                _st.make(r, t, progress_bar=progress)

def get_name(alternate_conf):
    n = f'{list(alternate_conf.keys())[0]}-{list(alternate_conf.values())[0]}'
    #     for s in '() ,':
    #         n = n.replace(s, '')
    return n


#!/usr/bin/env python

"""Shared common methods for reprocessing, not useful in itself"""

import argparse
import configparser
import importlib
import logging
import os
import grp
import json
import typing
import inspect

reprox_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

if 'REPROX_CONFIG' in os.environ:
    config_path = os.environ['REPROX_CONFIG']
else:
    config_path = os.path.join(reprox_dir, 'reprocessing.ini')
    print(f'Using {config_path}-config. Overwrite by setting "REPROX_CONFIG" '
          f'as an environment variable')

if not os.path.exists(config_path):
    raise FileNotFoundError(f'{config_path} does not exist')

config = configparser.ConfigParser()
config.sections()
config.read(config_path)
logging.basicConfig(
    level=getattr(logging, config['processing']['logging_level'].upper()),
    format=('%(asctime)s '
            '| %(name)-12s '
            '| %(levelname)-8s '
            '| %(message)s '
            '| %(funcName)s (l. %(lineno)d)'
            ),
    datefmt='%m-%d %H:%M')

log = logging.getLogger('reprocessing')

command = """
cd {base_folder}
straxer \
    {run_name} \
    --target {target} \
    --context {context} \
    --package {package} \
    --timeout {timeout} 
    {extra_options}
echo Processing job ended
"""

log_folder = os.path.join(config['context']['base_folder'], 'job_logs')
log_fn = os.path.join(log_folder, '{run_id}.txt')
runs_csv = os.path.join(config['context']['base_folder'], config['context']['runs_to_do'])

if not os.path.exists(os.path.split(log_fn)[0]):
    os.mkdir(os.path.split(log_fn)[0])


def get_context(package=config['context']['package'],
                context=config['context']['context'],
                output_folder=os.path.join(config['context']['base_folder'], 'strax_data'),
                config_kwargs: typing.Union[None, dict] = None,
                _minimum_run_number=int(config['context']['minimum_run_number']),
                _maximum_run_number=None,
                ):
    module = importlib.import_module(f'{package}.contexts')
    st = getattr(module, context)(output_folder=output_folder,
                                  minimum_run_number=_minimum_run_number,
                                  maximum_run_number=_maximum_run_number,
                                  )
    if config_kwargs is not None:
        log.warning(f'Updating the context with the following config {config_kwargs}')
        st.set_config(config_kwargs)
    st.context_config['check_available'] = []
    return st


def parse_args(description='nton reprocessing on dali',
               include_find_args=False,
               include_processing_args=False,
               include_workflow_args=False,
               ):
    """Parse arguments to return to the user"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--package',
        default=config['context']['package'],
        choices=['straxen', 'cutax'],
        type=str,
        help="Package to get context from"
    )
    parser.add_argument(
        '--context',
        default=config['context']['context'],
        type=str,
        help="Name of the context (should be in the package specified with --package)"
    )
    parser.add_argument(
        '--context-kwargs', '--context_kwargs', '--config',
        dest='context_kwargs',
        type=json.loads,
        default=None,
        help='Overwrite st.config settings using a json file. For example:'
             '--context_kwargs '
             '\'{'
             '"s1_min_coincidence": 2,'
             '"s2_min_pmts": 10'
             '}\''
    )
    parser.add_argument(
        '--config-kwargs', '--config_kwargs',
        dest='context_config_kwargs',
        type=json.loads,
        default={},
        help='overwrite st.context_config settings using a json file. For example:'
             '--config-kwargs '
             '\'{'
             '"output_folder": "./strax_data",'
             '}\''
    )
    parser.add_argument(
        '--targets', '--target',
        default=['event_info', 'event_pattern_fit'],
        nargs='*',
        help='Target final data type to produce. Can be a list for multicore mode.'
    )
    parser.add_argument(
        '--force-non-admin', '--force_non_admin',
        action='store_true',
        dest='force_non_admin',
        help='Allow non admin users to use this script.'
    )
    if include_find_args:
        parser = _include_find_args(parser)
    if include_processing_args:
        parser = _include_processing_args(parser)
    if include_workflow_args:
        parser = _include_workflow_args(parser)

    args = parser.parse_args()
    if hasattr(args, 'cmt_version') and args.cmt_version == 'False':
        args.cmt_version = False
    if not args.force_non_admin and not check_user_is_admin():
        raise PermissionError(
            f'{os.getlogin()}, you are not an admin so you probably don\'t'
            f' want to do a full reprocessing. In case you know what you are'
            f' doing add the "--force-non-admin" flag to you instructions')
    return args


def _include_find_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for finding data to the parser"""
    parser.add_argument(
        '--detectors', '--consider_detectors', '--consider-detectors',
        default=['tpc'],
        nargs='*',
        help='Data of detectors to process, choose one or more of "tpc, neutron_veto, muon_veto"'
    )
    parser.add_argument(
        '--cmt-version', '--cmt_version', '--check_cmt_version', '--check-cmt-version',
        default=config['context']['cmt_version'],
        type=str,
        dest='cmt_version',
        help='Specify CMT version if we should exclude runs that cannot be '
             '(fully) processed with this CMT version. Set to False if you '
             'don\'t want to run this check'
    )
    return parser


def _include_workflow_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for running the entire workflow to the parser"""
    parser.add_argument(
        '--move-after-workflow', '--move_after_workflow', '--move',
        action='store_true',
        dest='move_after_workflow',
        help='After running the workflow, move the data into the production folder'
    )
    parser.add_argument(
        '--skip-find', '--skip_find',
        action='store_true',
        dest='skip_find',
        help='If set, skip finding the data and just use the CSV file also previously used.'
    )
    return parser


def _include_processing_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for processing data to the parser"""
    parser.add_argument(
        '--ram', '--RAM',
        default=config['processing']['ram'],
        type=int,
        help='RAM [MB] per CPU to request'
    )
    parser.add_argument(
        '--cpu', '--cpu_per_job', '--cpu-per-job',
        default=config['processing']['cpus_per_job'],
        type=int,
        help='Number of CPUs per job to request'
    )
    parser.add_argument(
        '--submit-only', '--submit_only',
        default=config['processing']['submit_only'],
        type=int,
        help='Limits the total number of jobs to submit. Useful for testing. '
    )
    parser.add_argument(
        '--tag', '--container',
        default=config['processing']['container_tag'],
        type=str,
        help='Container to use for the reprocessing. '
    )
    parser.add_argument(
        '--clear_logs', '--clear-logs',
        dest='clear_logs',
        action='store_true',
        help='When submitting new jobs, first clear the logs'
    )
    return parser


def check_user_is_admin(admin_group='xenon1t-admins'):
    """Check that the user is an xenon1t-admin"""
    return admin_group in [grp.getgrgid(g).gr_name for g in os.getgroups()]


def log_versions():
    """Log versions (nested import makes the arg parsing quick)"""
    import straxen
    log.warning(straxen.print_versions(return_string=True),
                )


if __name__ == '__main__':
    raise ValueError('core.py is not run on it\'s own, you are looking for run_workflow.py instead')




name = '2022_02_03_lowe'
fax_file = 'fax_config_nt_sr0_v0.json'
base_dir = '/dali/lgrandi/angevaare/wfsims/pema'

# Fixed
data_name = f'pema_w{wfsim.__version__}_p{pema.__version__}'
fig_dir = os.path.join(base_dir, name, f'figures_summary_{data_name}')
data_dir = os.path.join(base_dir, name, 'processed_data')
raw_data_dir = os.path.join(base_dir, name, 'raw_data')
instructions_csv = f"./inst_{data_name}_{name}.csv"


# You need this for setting up the dali-jobs
environ_init = '''eval "$(/home/angevaare/software/Miniconda3/bin/conda shell.bash hook)"
conda activate strax
export PATH=/home/angevaare/software/Miniconda3/envs/strax/bin:$PATH'''

# Output naming
default_label = 'Normal clustering'
custom_label = 'Changed clustering'

# Take a few arbitrary runs that allow to run jobs in parallel and get the
# gains from CMT
run_list = list(f'{r:06}' for r in range(27000, 27000+5))

print(run_list)
# Just some id which allows CMT to load
run_id = run_list[0]

# setting up instructions like this may take a while. You can set e.g.
instructions = dict(
    event_rate=20,  # Don't make too large -> overlapping truth info
    chunk_size=5,  # keep large -> less overhead but takes more RAM
    n_chunk=100, # back to 200
    tpc_radius=straxen.tpc_r,
    tpc_length=straxen.tpc_z,

    energy_range=[1,5], # keV
    nest_inst_types = wfsim.NestId.ER,
)

event_inst = pema.rand_instructions(**instructions)

peak_instructions = {
    k: instructions[k]
    for k in
    'tpc_radius tpc_length drift_field nest_inst_types'.split()
}
peak_instructions.update(dict(n_peaks=5_000,
                              s1_amplitude_range=[1, 100],
                              s2_amplitude_range=[1, 1000],
                              time_separation_ns=1e7,
                              min_time_seperation=100e3,
                              start_time=np.max(event_inst['time']),
                              ))
peak_inst = pema.random_peaks(**peak_instructions)[:0]
df = pd.DataFrame(np.concatenate([event_inst, peak_inst]))
df = df[df['amp'] > 0]
df.to_csv(instructions_csv, index=False)



config_update = dict(
    detector='XENONnT',
    fax_file=os.path.abspath(instructions_csv),
    fax_config=fax_file,
    fdc_map="XnT_3D_FDC_xyt_dummy_all_zeros_v0.1.json.gz",
    fax_config_override={
        'enable_electron_afterpulses':True,
        'enable_pmt_afterpulses':True,
        'enable_noise': True,
    })



    def test_later_make_ev_matched(self):
        st = self.script.st
        st.get_array(run_id, 'truth_events')

    def test_later_rec_bas(self):
        st = self.script.st
        st2 = st.new_context()
        peaks_1 = st.get_array(run_id, 'match_acceptance_extended')
        peaks_2 = st2.get_array(run_id, 'match_acceptance_extended')
        peaks_1_kwargs = dict(bins=10)
        if len(peaks_1):
            pema.summary_plots.rec_plot(peaks_1, **peaks_1_kwargs)
            pema.summary_plots.reconstruction_bias(peaks_1, s1_kwargs=peaks_1_kwargs)
            plt.clf()
        if len(peaks_1) and len(peaks_2):
            if not np.sum(peaks_1['type'] == 1):
                return
            pema.summary_plots.rec_diff(
                peaks_1,
                peaks_2,
                s1_kwargs=peaks_1_kwargs,
                s2_kwargs=peaks_1_kwargs,
            )
            plt.clf()

    def test_later_inst_plot(self):
        st = self.script.st
        peaks = st.get_array(run_id, 'peaks')
        st.plot_peaks(run_id, time_within=peaks[0])
        plt.clf()
        st.plot_peaks(run_id, time_within=peaks[0], xaxis='since_start')
        plt.clf()
        st.plot_peaks(run_id, time_within=peaks[0], xaxis=False)
        pema.save_canvas('test_fig', os.path.join(self.tempdir, 'figs'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)