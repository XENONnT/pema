"""Helpers for doing processing on dali"""
import os
import json
import strax
import typing as ty
from immutabledict import immutabledict
import subprocess
from collections import defaultdict
import pandas

job_script = """\
#!/bin/bash
#SBATCH --partition {partition}
#SBATCH --qos {partition}
#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={max_hours}

echo Processing job started

# Activate your environment you want to use
{bash_activate}

echo Environment activated
{cmd}
echo Processing job ended
"""


def write_script(fn, script, **kwargs):
    with open(fn, mode='w') as f:
        f.write(script.format(**kwargs))
    mode = os.stat(fn).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(fn, mode)


def write_dict_to_json(path: str,
                       to_write: dict, ):
    assert path.endswith('json')
    json_options = dict(sort_keys=True, indent=4)
    with open(path, mode='w') as f:
        f.write(json.dumps(to_write, **json_options))


class ProcessRun:
    log_file = None
    script_file = None
    base_dir_requires = ('configs', 'logs', 'scripts')

    def __init__(self,
                 st: strax.context,
                 run_id: ty.Union[str, tuple, list],
                 target: ty.Union[str, tuple],
                 config: ty.Union[dict, immutabledict, None] = None,
                 ):
        self.st = st.new_context()
        self.run_id = strax.to_str_tuple(run_id)
        self.target = strax.to_str_tuple(target)
        if config is None:
            config = {}
        self.config = config
        self.st.set_config(config)

        self.base_dir = self.extract_base_dir(st)
        for subdir in self.base_dir_requires:
            os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)

    def __repr__(self):
        repr = f'ProcessRun {self.run_id} - {self.target}:\n{self.config}'
        repr += f'\nwrite to {self.log_file} from {self.script_file}'
        return repr

    def all_stored(self, show_key=False, return_bool=False):
        bool_stored = True
        res = defaultdict(list)
        for r in self.run_id:
            res['number'].append(r)
            for t in self.target:
                stored = self.st.is_stored(r, t)
                bool_stored &= stored
                res[t].append(stored)
                if show_key:
                    res[f'{t}-key'] = self.st.key_for(r, t)

        df = pandas.DataFrame(res)
        df.set_index('number')
        if return_bool:
            return bool_stored
        return df

    @staticmethod
    def extract_base_dir(st):
        """Extract a base dir from the context (either from self or storage)"""
        if hasattr(st, 'base_dir'):
            base_dir = st.base_dir
        else:
            base_dir = None
            for sf in st.storage:
                if hasattr(sf, 'path'):
                    base_dir = sf.path
                    break
        if base_dir is None:
            raise FileNotFoundError('Cannot continue, no basedir')
        return base_dir

    def _fmt(self, directory: str, name: str) -> str:
        return os.path.join(self.base_dir, directory, name)

    def make_cmd(self,
                 debug=True):
        """
        return_command = just return the command, don't do the actual file stuf
        """
        st = self.st
        tot_config = st.config.copy()
        if 'channel_map' in tot_config:
            # Not saveable to JSON
            del tot_config['channel_map']

        run_id = self.run_id[0]
        target = self.target[0]
        this_key = st.key_for(run_id, target)
        if len(self.run_id) > 1:
            job_name = f'{run_id}_{self.run_id[-1]}_{this_key}'
        else:
            job_name = f'{this_key}'
        if len(self.target) > 1:
            job_name += '-'.join(self.target[1:])

        conf_file = self._fmt('configs', f'config_{job_name}.json')

        cmd = (f'pema_straxer {" ".join(self.run_id)} '
               f'--target {" ".join(self.target)} '
               f'--context pema_context '
               f'--init_from_json {conf_file} '
               )

        if len(st.storage) == 1:
            raw_dir = None
            data_dir = st.storage[0].path
        elif len(st.storage) == 2:
            raw_dir = st.storage[0].path
            data_dir = st.storage[1].path
        else:
            raise ValueError
        context_init = dict(base_dir=self.base_dir,
                            config_update=tot_config,
                            raw_dir=raw_dir,
                            data_dir=data_dir,
                            )
        if debug:
            cmd += ' --debug'
        if not sum([st.is_stored(run_id, 'raw_records') for run_id in self.run_id]):
            cmd += ' --build_lowlevel --rechunk_rr'

        write_dict_to_json(conf_file, context_init)
        return cmd, job_name

    def exec_dali(self,
                  cmd,
                  job_name,
                  bash_activate,
                  mem=2000,
                   partition='xenon1t',
                   max_hours="04:00:00"
                   ):
        self.log_file = self._fmt('logs', f'{job_name}.log')
        self.script_file = self._fmt('scripts', f'{job_name}.sh')
        script = job_script.format(
            bash_activate=bash_activate,
            log_file=self.log_file,
            cmd=cmd,
            mem=mem,
            partition=partition,
            max_hours=max_hours)
        write_script(self.script_file, script)
        cp = subprocess.run(f'sbatch {self.script_file}', shell=True, capture_output=True)
        self.job_id = cp.stdout
        return self.job_id

    def exec_local(self, cmd, job_name):
        self.log_file = self._fmt('logs', f'{job_name}.log')
        self.script_file = f'{cmd} &>{self.log_file}'
        # subprocess.Popen(self.script_file.split(' '), shell=True)
        # cp = subprocess.run(self.script_file, shell=True, capture_output=True)
        # return cp
        cmd = cmd.replace('  ', ' ')
        p = subprocess.Popen(cmd.split(' '),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
        self.log_file = p

    def read_log(self):
        if self.log_file is None:
            raise RuntimeError('No logfile')
        f = open(self.log_file, "r")
        lines = [l for l in f]
        f.close()
        return lines

    def job_finished(self):
        finished = False
        for line in self.read_log()[-10:]:
            if 'Error' in line:
                raise ValueError(line)
            elif 'ended' in line:
                finished = True
        return finished

