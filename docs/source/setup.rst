Setting up Pema
===================

Install using:
`pip install pema`

For the true power of pema, database access is assumed.
This can be achieved by setting up a utilix file with the proper passwords.
Please follow the installation guide as from straxen:
https://straxen.readthedocs.io/en/latest/setup.html


Basic examples
--------------
Imagine you want to load peaks from ``wfsim`` and check the properly matched peaks. We assume one is familiar with  You can do so as follows:


.. code-block:: python
  
  import pema
  import numpy as np
  
  # Setup a simulation, see e.g. pema/tests or wfsim documentation
  st_wfsim = pema.contexts.pema_context(..)
  
  # Load peaks and truth information  
  peaks = st_wfsim.get_array(run_id, ('peak_basics', 'peak_id'))  # Same dtype
  truths = st_wfsim.get_array(run_id, 'match_acceptance_extended')  # Super type of truth

  truth_are_matched = truths['outcome']=='found'  # This means that the peak as in the truth was correctly found
  peaks_are_machted = np.in1d(peaks['id'], truths[truth_are_matched]['matched_to'])

  assert len(peaks_are_matched) == len(peaks)

  peaks_with_good_match = peaks[peaks_are_machted]
  truths_with_good_match = truths[truth_are_matched]

Similar selections can easily be extended by setting e.q. ``truth_are_matched = truths['outcome']=='missed'`` etc.
