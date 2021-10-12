import numpy as np
import pandas as pd
import wfsim
import straxen
import strax
import nestpy
import typing as ty
from strax.utils import tqdm
from warnings import warn

export, __all__ = strax.exporter()


@export
def rand_instructions(
        event_rate: int,
        chunk_size: int,
        n_chunk: int,
        drift_field: float,
        energy_range: ty.Union[tuple, list, np.ndarray],
        tpc_length: float = straxen.tpc_z,
        tpc_radius: float = straxen.tpc_r,
        nest_inst_types: ty.Union[ty.List[int], ty.Tuple[ty.List], np.ndarray, None] = None,
) -> dict:
    """
    Generate instructions to run WFSim
    :param event_rate: # events per second
    :param chunk_size: the size of each chunk
    :param n_chunk: the number of chunks
    :param energy_range: the energy range (in keV) of the recoil type
    :param tpc_length: the max depth of the detector
    :param tpc_radius: the max radius of the detector
    :param nest_inst_types: the
    :param drift_field:
    :return:
    """
    if nest_inst_types is None:
        nest_inst_types = [7]

    n_events = event_rate * chunk_size * n_chunk
    total_time = chunk_size * n_chunk

    inst = np.zeros(2 * n_events, dtype=wfsim.instruction_dtype)
    inst[:] = -1

    uniform_times = total_time * (np.arange(n_events) + 0.5) / n_events

    inst['time'] = np.repeat(uniform_times, 2) * int(1e9)
    inst['event_number'] = np.digitize(inst['time'],
                                       1e9 * np.arange(n_chunk) *
                                       chunk_size) - 1
    inst['type'] = np.tile([1, 2], n_events)

    r = np.sqrt(np.random.uniform(0, tpc_radius ** 2, n_events))
    t = np.random.uniform(-np.pi, np.pi, n_events)
    inst['x'] = np.repeat(r * np.cos(t), 2)
    inst['y'] = np.repeat(r * np.sin(t), 2)
    inst['z'] = np.repeat(np.random.uniform(tpc_length, 0, n_events), 2)

    # Here we'll define our XENON-like detector
    nc = nestpy.NESTcalc(nestpy.VDetector())
    A = 131.293
    Z = 54.
    density = 2.862  # g/cm^3   #SR1 Value

    energy = np.random.uniform(*energy_range, n_events)
    quanta = []
    exciton = []
    recoil = []
    for en in tqdm(energy, desc='generating from nest'):
        interaction_type = np.random.choice(nest_inst_types)
        interaction = nestpy.INTERACTION_TYPE(interaction_type)
        y = nc.GetYields(interaction,
                         en,
                         density,
                         drift_field,
                         A,
                         Z,
                         )
        q = nc.GetQuanta(y, density)
        quanta.append(q.photons)
        quanta.append(q.electrons)
        exciton.append(q.excitons)
        exciton.append(0)
        # both S1 and S2
        recoil += [interaction_type, interaction_type]

    inst['amp'] = quanta
    inst['local_field'] = drift_field
    inst['n_excitons'] = exciton
    inst['recoil'] = recoil
    for field in inst.dtype.names:
        if np.any(inst[field] == -1):
            warn(f'{field} is not (fully) filled')
    return inst


@export
def kr83_instructions(input_inst: dict,
                      z_max=-148.1,
                      r_max=straxen.tpc_r) -> dict:
    """
    Given instructions in a dict (first arg) generate the instructions
    that can be fed to wfsim. Will generate a KR-like dataset!
    :param input_inst: dict of
    :param z_max: max depth of interactions in TPC
    :param r_max: max radius of interactions in TPC
    :return: dict with filled instructions
    """
    warn('Deprecated! Should be updated before use')
    # Uses Peters example to generate KR-like data. T
    import nestpy
    half_life = 156.94e-9  # Kr intermediate state half-life in ns
    decay_energies = [32.2, 9.4]  # Decay energies in kev

    n = input_inst['nevents'] = input_inst['event_rate'] * input_inst['chunk_size'] * input_inst[
        'nchunk']
    input_inst['total_time'] = input_inst['chunk_size'] * input_inst['nchunk']

    instructions = np.zeros(4 * n, dtype=wfsim.instruction_dtype)
    instructions['event_number'] = np.digitize(
        instructions['time'],
        1e9 * np.arange(input_inst['nchunk']) * input_inst['chunk_size']) - 1

    instructions['type'] = np.tile([1, 2], 2 * n)
    # TODO fix this 7
    instructions['recoil'] = [7 for i in range(4 * n)]

    r = np.sqrt(np.random.uniform(0, r_max ** 2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 4)
    instructions['y'] = np.repeat(r * np.sin(t), 4)
    instructions['z'] = np.repeat(np.random.uniform(z_max, 0, n), 4)

    # To get the correct times we'll need to include the 156.94 ns half
    # life of the intermediate state.
    if input_inst.get('timing', 'uniform') == 'uniform':
        uniform_times = input_inst['total_time'] * (np.arange(n) + 0.5) / n
    elif input_inst['timing'] == 'increasing':
        uniform_times = input_inst['total_time'] * np.sort(
            np.random.triangular(0, 0.9, 1, n))
    else:
        timing = input_inst['timing']
        raise ValueError(f'Timing {timing} unknown, Choose "uniform" or "increasing"')
    delayed_times = uniform_times + np.random.exponential(half_life / np.log(2), len(uniform_times))
    instructions['time'] = np.repeat(list(zip(uniform_times, delayed_times)), 2) * 1e9

    # Here we'll define our XENON-like detector
    nc = nestpy.NESTcalc(nestpy.VDetector())
    A = 131.293
    Z = 54.
    density = input_inst.get('density', 2.862)  # g/cm^3   #SR1 Value
    drift_field = input_inst.get('drift_field', 82)  # V/cm    #SR1 Value
    interaction = nestpy.INTERACTION_TYPE(7)

    energy = np.tile(decay_energies, n)
    quanta = []
    for en in energy:
        y = nc.GetYields(interaction,
                         en,
                         density,
                         drift_field,
                         A,
                         Z,
                         (1, 1))
        quanta.append(nc.GetQuanta(y, density).photons)
        quanta.append(nc.GetQuanta(y, density).electrons)

    instructions['amp'] = quanta
    return instructions


@export
def inst_to_csv(csv_file: str,
                get_inst_from=rand_instructions,
                **kwargs):
    """
    Write instructions to csv file
    :param csv_file: path to the csv file
    :param get_inst_from: function to modify instructions and generate
    S1 S2 instructions from
    :param kwargs: key word arguments to give to the get_inst_from-function
    """
    pd.DataFrame(get_inst_from(**kwargs)).to_csv(csv_file, index=False)
