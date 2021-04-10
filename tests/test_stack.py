import pema
import os
import strax
import straxen
import wfsim
import os
import pema
import time
import tempfile
straxen.print_versions(['strax', 'straxen', 'wfsim', 'nestpy', 'pema'])

run_id = '008000'


class RunSim:
    def __init__(self):
        # setting up instructions like this may take a while. You can set e.g.
        self.instructions = dict(
            event_rate=1,
            chunk_size=1,
            nchunk=1,
            photons_low=1,
            photons_high=100,
            electrons_low=1,
            electrons_high=100,
            tpc_radius=straxen.tpc_r,
            tpc_length=148.1,
            drift_field=18.5,
            timing='uniform',
        )

        print(f'Init done')

    def run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                pema.inst_to_csv(
                    self.instructions,
                    instructions_csv := os.path.join(temp_dir, 'inst.csv'),
                    get_inst_from=pema.rand_instructions)

                config_update = {
                    "detector": 'XENONnT',
                    "fax_file": os.path.abspath(instructions_csv),
                    "fax_config": 'fax_config_nt_low_field.json',
                }

                print("Temporary directory is ", temp_dir)
                os.chdir(temp_dir)

                st = pema.pema_context(base_dir = temp_dir,
                                       raw_dir = temp_dir,
                                       data_dir = temp_dir,
                                       config_update=config_update,)
                st.set_context_config(
                    {'allow_shm': True,
                     'allow_lazy': False,
                     'timeout': 300,
                     'max_messages': 10,
                     }
                )
                print(f'Start script')
                script_writer = pema.ProcessRun(st, run_id,
                                                ('raw_records', 'records',
                                                 'peaklets', 'peaks_matched',
                                                 'match_acceptance_extended'
                                                 ))

                cmd, name = script_writer.make_cmd()
                ret = script_writer.exec_local(cmd, name)
                print(f'Starting\n\t{cmd}')
                print(script_writer.log_file.communicate())
                time.sleep(10)
                print(f'Done')
                print(f'Stored: {script_writer.all_stored()}')
                assert script_writer.all_stored(return_bool=True)
            # On windows, you cannot delete the current process'
            # working directory, so we have to chdir out first.
            finally:
                os.chdir('..')


def test_run():
    if not straxen.utilix_is_configured():
        return
    sim = RunSim()
    sim.run()
