def test_import():
    import pema

    from pema import matching
    from pema import MatchPeaks
    from pema import append_fields
    try:
        from pema import _plot_peak_matching_histogram
        raise ValueError(
            '_plot_peak_matching_histogram should not be available at top level')
    except (ImportError, ValueError, FileNotFoundError, ModuleNotFoundError):
        # Good we cannot import this (which is what we want)
        pass
    print('done')
