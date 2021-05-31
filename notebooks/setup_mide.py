import pema
import os
import straxen
import wfsim

name = 'mide'
base_dir = '/dali/lgrandi/angevaare/wfsims/pema'

# Fixed
data_name = f'pema_w{wfsim.__version__}_p{pema.__version__}'
fig_dir = os.path.join(base_dir, f'figures_summary_{data_name}')
data_dir = os.path.join(base_dir, name, 'processed_data')
raw_data_dir = os.path.join(base_dir, 'raw_data')
instructions_csv = f"./inst_{data_name}.csv"

# You need this for setting up the dali-jobs
environ_init = '''eval "$(/home/angevaare/software/Miniconda3/bin/conda shell.bash hook)"
conda activate strax
export PATH=/home/angevaare/software/Miniconda3/envs/strax/bin:$PATH'''

# Output naming
default_label = 'Normal clustering'
custom_label = 'Changed clustering'

# Take a few arbitrary runs that allow to run jobs in parallel and get the 
# gains from CMT
run_list = list(f'{r:06}' for r in range(18000, 18000 + 20))

# Just some id which allows CMT to load
run_id = run_list[0]

# setting up instructions like this may take a while. You can set e.g. 
instructions = dict(
    event_rate=2,
    chunk_size=5,
    nchunk=100,  # 40 works
    photons_low=1,  # PE
    photons_high=1e4,  # PE
    electrons_low=1,  #
    electrons_high=1e4,
    tpc_radius=straxen.tpc_r,
    tpc_length=straxen.tpc_z,
    drift_field=straxen.get_resource('fax_config_nt_low_field.json', fmt='json').get('drift_field'),
    timing='uniform',  # Double S1 peaks uniform over time
)

pema.inst_to_csv(
    instructions,
    instructions_csv,
    get_inst_from=pema.rand_instructions)

# TODO can we add noise?
config_update = dict(
    detector='XENONnT',
    fax_file=os.path.abspath(instructions_csv),
    fax_config='fax_config_nt_low_field.json',
    #     fax_config_override=dict(enable_electron_afterpulses=False,
    #                              enable_pmt_afterpulses=False
    #                             ),
)
