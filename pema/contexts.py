import strax
import straxen
import wfsim
import pema
import os

export, __all__ = strax.exporter()


@export
def pema_context(
        base_dir: str,
        config_update: dict = None,
        raw_dir=None,
        data_dir=None,
        raw_types=None,
):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f'Cannot use {base_dir} as base_dir. It does not exist.')
    config = dict(detector='XENONnT',
                  check_raw_record_overlaps=False,
                  fax_config='fax_config_nt_low_field.json',
                  **straxen.contexts.xnt_common_config
                  )
    if config_update is not None:
        if not isinstance(config_update, dict):
            raise ValueError(f'Invalid config update {config_update}')
        config = strax.combine_configs(config, config_update)
    st = strax.Context(
        config=config,
        **straxen.contexts.common_opts)
    st.set_config(config)
    # Disable warning for these two options
    st.set_context_config({'free_options': ('n_nveto_pmts', 'channel_map')})

    st.register(wfsim.RawRecordsFromFaxNT)
    st.register_all(pema.match_plugins)
    st.register_all(straxen.plugins.position_reconstruction)
    del st._plugin_class_registry['peak_positions_base_nt']

    if raw_types is None:
        raw_types = (wfsim.RawRecordsFromFaxNT.provides +
                     straxen.plugins.pulse_processing.PulseProcessing.provides)

    st.storage = []

    if raw_dir is not None:
        st.storage += [strax.DataDirectory(
            raw_dir,
            take_only=raw_types)]
    if data_dir is not None:
        st.storage += [strax.DataDirectory(
            data_dir,
            exclude=raw_types
        )]
    if not len(st.storage):
        raise RuntimeError('No storage, provide raw_dir and/or data_dir')

    return st
