import time
import strax
import straxen
import pema
import wfsim
import os
from warnings import warn
import configparser


def get_pema_config(config_location: str=None):
    if config_location is None:
        config_location=os.path.join(os.path.abspath('.'), 'pema.ini')
        warn(f'No config_location specified, assuming {config_location}', UserWarning)
    ini_file = configparser.ConfigParser()
    ini_file.sections()
    if not os.path.exists(config_location):
        raise FileNotFoundError(config_location)
    ini_file.read(config_location)

    config= dict()
    config['context_config']=dict(ini_file._sections['context'])
    config['context_config']['fax_config_override'] = dict(ini_file._sections['fax_config_override'])
    config['simulation']=dict(ini_file._sections['simulation'])
    config['submission'] = dict(ini_file._sections['submission'])
    return config


def write_subdirs(config):
    for target_dir in 'fig_dir data_dir raw_data_dir instructions logs scripts'.split():
        destination = os.path.join(config['simulation']['base_dir'], target_dir)
        config['simulation'][target_dir] = destination
        os.makedirs(destination, exist_ok=True)
    return config


class PemaRunner:
    jobs = None
    config = {}

    def __init__(self, config: dict):
        config = write_subdirs(config)
        self.config = config

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]
        for config in ['simulation', 'context_config']:
            if item in self.config[config]:
                return self.config[config][item]

        return super().__getattribute__(item)

    @property
    def drift_field(self):
        return straxen.get_resource(self.fax_config, fmt='json').get('drift_field')

    def get_context(self, run_id):
        # setting up instructions like this may take a while. You can set e.g.
        interaction_type = (wfsim.NestId.ER
                            if self.interaction_types == 'wfsim.NestId.ER'
                            else int(self.interaction_types))
        instructions = dict(
            event_rate=int(self.event_rate),  # Don't make too large -> overlapping truth info
            chunk_size=int(self.chunk_size),  # keep large -> less overhead but takes more RAM
            n_chunk=int(self.n_chunk),
            tpc_radius=straxen.tpc_r,
            tpc_length=straxen.tpc_z,
            drift_field=self.drift_field,
            energy_range=[float(self.energy_min), float(self.energy_max)],  # keV
            nest_inst_types=interaction_type
        )
        instructions_csv = os.path.join(self.instructions,
                                        f'{run_id}_{strax.deterministic_hash(instructions)}.csv'
                                        )
        pema.inst_to_csv(
            instructions_csv,
            get_inst_from=pema.rand_instructions,
            **instructions)

        kwargs = straxen.filter_kwargs(pema.pema_context,
                                       {**self.context_config,
                                        **self.simulation,
                                        }
                                       )
        st = pema.pema_context(config_update=dict(**self.context_config,
                                                  fax_file=instructions_csv,),
                               **kwargs)
        return st

    @property
    def run_ids(self):
        cmt_id = int(self.context_config['cmt_run_id_sim'])
        runs = range(cmt_id, cmt_id + int(self.number_of_runs))
        run_ids = [f'{r:06}' for r in runs]
        return run_ids

    def submit(self):
        jobs = []
        for run_id in self.run_ids:
            jobs += [
                pema.ProcessRun(
                    self.get_context(run_id),
                    run_id=run_id,
                    target=('raw_records',
                            'peaks',
                            'event_info',
                            'truth_matched',
                            'match_acceptance_extended'),
                )
            ]
        for job in jobs:
            cmd, job_name = job.make_cmd()
            if self.submission.get('_run_with_subprocess', False):
                job.exec_local(cmd, job_name)
            else:
                job.exec_dali(
                    cmd,
                    job_name,
                    mem=self.submission["ram"],
                    container=f'xenonnt-{self.submission["tag"]}.simg')
        self.jobs = jobs

    def finish(self):
        if self.jobs is None:
            return
        while True:
            n_running = sum([not j.job_finished for j in self.jobs])
            if n_running == 0:
                break
            else:
                print(f'Running {n_running}, sleep')
                time.sleep(1)

if __name__ == '__main__':
    config = get_pema_config('/mnt/c/google_drive/PhD-master/ubuntu-storage/ubuntu-windows/software/pema/bin/pema.ini')
    print(config)
    runner = PemaRunner(config)
    print(runner.run_ids)
    print(runner.get_context(runner.run_ids[0]).config['fax_config_override'])
    runner.submit()
    runner.finish()

