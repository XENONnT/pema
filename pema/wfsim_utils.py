import numpy as np
import pandas as pd
import wfsim
import straxen
import strax

export, __all__ = strax.exporter()

instruction_dtype = [('event_number', np.int), ('type', np.int), ('time', np.int64),
                     ('x', np.float32), ('y', np.float32), ('z', np.float32),
                     ('amp', np.int), ('recoil', '<U2')]


@export
def rand_instructions(input_inst: dict,
                      z_max=-148.1,
                      r_max=straxen.tpc_r) -> dict:
    """
    Given instructions in a dict (first arg) generate the instructions
    that can be fed to wfsim
    :param input_inst: dict of
    :param z_max: max depth of interactions in TPC
    :param r_max: max radius of interactions in TPC
    :return: dict with filled instructions
    """
    n = input_inst['nevents'] = input_inst['event_rate'] * input_inst[
        'chunk_size'] * input_inst['nchunk']
    input_inst['total_time'] = input_inst['chunk_size'] * input_inst['nchunk']

    inst = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = input_inst['total_time'] * (np.arange(n) + 0.5) / n

    inst['time'] = np.repeat(uniform_times, 2) * int(1e9)
    inst['event_number'] = np.digitize(inst['time'],
                                       1e9 * np.arange(input_inst['nchunk']) *
                                       input_inst['chunk_size']) - 1
    inst['type'] = np.tile([1, 2], n)
    # TODO fix this 7
    inst['recoil'] = [7 for i in range(n * 2)]

    r = np.sqrt(np.random.uniform(0, r_max ** 2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    inst['x'] = np.repeat(r * np.cos(t), 2)
    inst['y'] = np.repeat(r * np.sin(t), 2)
    inst['z'] = np.repeat(np.random.uniform(z_max, 0, n), 2)

    # If some of these things are in the instructions, extract them. Otherwise
    # use the default values
    photons_low = input_inst.get('photons_low', 3)
    photons_high = input_inst.get('photons_high', 100)
    electrons_low = input_inst.get('electrons_low', 10)
    electrons_high = input_inst.get('electrons_high', 1000)

    nphotons = np.random.randint(photons_low, photons_high + 1, n)
    nelectrons = np.random.randint(electrons_low, electrons_high + 1, n)
    inst['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(int)

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
def inst_to_csv(instructions: dict,
                csv_file: str,
                get_inst_from=rand_instructions,
                **kwargs):
    """
    Write instructions to csv file
    :param instructions: Instructions to start with
    :param csv_file: path to the csv file
    :param get_inst_from: function to modify instructions and generate
    S1 S2 instructions from
    :param kwargs: key word arguments to give to the get_inst_from-function
    """
    pd.DataFrame(get_inst_from(instructions, **kwargs)).to_csv(csv_file, index=False)
