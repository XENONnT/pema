import pema


def test_make_kr_isnt():
    input_dict = dict(
        nevents=1,
        event_rate=1,
        chunk_size=1,
        nchunk=1,
    )
    pema.kr83_instructions(input_dict)
