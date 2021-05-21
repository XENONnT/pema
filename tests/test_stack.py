import pema
import os
import strax
import straxen
import wfsim
import os
import pema
import time
import tempfile
import matplotlib.pyplot as plt
import strax
from strax.testutils import Records, Peaks, run_id
import tempfile
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
import typing as ty
import os
import unittest
import shutil
import uuid

straxen.print_versions(['strax', 'straxen', 'wfsim', 'nestpy', 'pema'])

run_id = '008000'


class TestStack(unittest.TestCase):
    """Test the per-run defaults options of a context"""

    @classmethod
    def setUpClass(cls):
        cls.set_temp()
        cls.set_script()

    @classmethod
    def set_temp(cls):
        temp_folder = uuid.uuid4().hex
        cls.tempdir = os.path.join(tempfile.gettempdir(), temp_folder)
        os.mkdir(cls.tempdir)

    @classmethod
    def set_script(cls):
        # setting up instructions like this may take a while. You can set e.g.
        instructions = dict(
            event_rate=1,
            chunk_size=1,
            nchunk=1,
            photons_low=10,
            photons_high=10,
            electrons_low=10,
            electrons_high=10,
            tpc_radius=straxen.tpc_r,
            tpc_length=148.1,
            drift_field=18.5,
            timing='uniform',
        )
        temp_dir = cls.tempdir
        instructions_csv = os.path.join(temp_dir, 'inst.csv')
        pema.inst_to_csv(
            instructions,
            instructions_csv,
            get_inst_from=pema.rand_instructions)

        config_update = {
            "detector": 'XENONnT',
            "fax_file": os.path.abspath(instructions_csv),
            "fax_config": 'fax_config_nt_low_field.json',
        }

        print("Temporary directory is ", temp_dir)
        os.chdir(temp_dir)

        st = pema.pema_context(base_dir=temp_dir,
                               raw_dir=temp_dir,
                               data_dir=temp_dir,
                               config_update=config_update, )
        st.set_context_config(
            {'allow_shm': True,
             'allow_lazy': False,
             'timeout': 300,
             'max_messages': 10,
             }
        )
        script_writer = pema.ProcessRun(st, run_id,
                                        ('raw_records', 'records',
                                         'peaklets', 'peaks_matched',
                                         'match_acceptance_extended'
                                         ))
        cls.script = script_writer

    def test_first_run_execute(self):
        print(f'Start script')

        cmd, name = self.script.make_cmd()
        self.script.exec_local(cmd, name)
        print(f'Starting\n\t{cmd}')
        t0 = time.time()
        print(self.script.process.communicate())
        print(f'took {time.time() - t0:.2f}s')
        time.sleep(10)

        print(f'Done')
        print(f'Stored: {self.script.all_stored()}')
        assert self.script.all_stored(return_bool=True)
        assert os.path.exists(self.script.log_file)
        if not self.script.job_finished():
            print(self.script.read_log())
            raise ValueError(f'Job did not finish')

    def test_first_run_plugins(self):
        self.script.purge_below('match_acceptance_extended')
        for t in strax.to_str_tuple(self.script.target):
            for r in strax.to_str_tuple(self.script.run_id):
                if (self.script.st._plugin_class_registry[t].save_when
                        > strax.SaveWhen.NEVER):
                    self.script.st.make(r, t)
                    assert self.script.st.is_stored(r, t)

    def test_later_compare(self):
        st = self.script.st
        st2 = st.new_context()
        for t in strax.to_str_tuple(self.script.target):
            print(run_id, t)
            st2.make(run_id, t)
        peaks_1 = st.get_array(run_id, 'match_acceptance_extended')
        peaks_2 = st2.get_array(run_id, 'match_acceptance_extended')
        if not 'run_id' in peaks_1.dtype.names:
            peaks_1 = pema.append_fields(peaks_1, 'run_id', [run_id] * len(peaks_1))
            peaks_2 = pema.append_fields(peaks_2, 'run_id', [run_id] * len(peaks_2))
        pema.compare_outcomes(st, peaks_1,
                              st2, peaks_2,
                              max_peaks=11,
                              show=False,
                              fig_dir=self.tempdir,
                              )
        plt.clf()

    def test_later_rec_bas(self):
        peaks = st.get_array(run_id, )
        pema.rec_plot(peaks, 'match_acceptance_extended',
                      **dict(bins=50, range=[[0, 1e6], [0, 10]]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)
