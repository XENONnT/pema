import pema

_input_dict = dict(
    nevents=1,
    event_rate=1,
    chunk_size=1,
    nchunk=1,
)


def test_make_kr_isnt():
    pema.kr83_instructions(_input_dict)


def test_rand_instructions():
    pema.rand_instructions(_input_dict)
