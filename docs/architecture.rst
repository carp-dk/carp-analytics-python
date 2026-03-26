Architecture
============

The package is intentionally split into small services:

* ``carp.study`` composes the public `CarpStudy` entrypoint.
* ``carp.participants`` handles participant parsing and lookup.
* ``carp.records`` streams and filters JSON records.
* ``carp.schema`` infers measurement schemas.
* ``carp.export`` writes JSON output and grouped files.
* ``carp.frames`` loads pandas dataframes and writes parquet files.
* ``carp.types`` generates dataclasses from sampled records.
* ``carp.plotting`` renders HTML maps for participant data.
