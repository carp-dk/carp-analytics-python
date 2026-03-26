CLI
===

The command line interface exposes the same core flows as the Python API.

.. code-block:: bash

   carp schema sleep-data/phase-1-1/data-streams.json
   carp count sleep-data/phase-1-1/data-streams.json
   carp participants sleep-data/phase-1-1/data-streams.json
   carp export sleep-data/phase-1-1/data-streams.json -o output.json -t dk.cachet.carp.stepcount
   carp group sleep-data/phase-1-1/data-streams.json -o grouped_output
   carp convert sleep-data/phase-1-1/data-streams.json -o output_parquet
