# Pseudorange differencing module for Baguette in the sky (bits-prd)

A module to compute baseline using single or double differences.

The module depends on [Baguette in the sky](https://github.com/enkisaura/Baguette-In-The-Sky).

---
## Usage
```python
import os
import bits

from bits_prd.src import steering_vectors, code_prd

# 1 Parse data from two receivers with corresponding ephemeris
data_filepath = os.path.join(os.getcwd(), "bits_prd", "test", "data")
# Raw data
rx1_raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
rx2_raw_filepath = os.path.join(data_filepath, "RX0200FRA_R_20261241729_00U_01S_MO.rnx")
# Ephemeris
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

# Parse
raw_rx1_pd = bits.parsers.gnss_raw.rinex_obs(rx1_raw_filepath)
raw_rx2_pd = bits.parsers.gnss_raw.rinex_obs(rx2_raw_filepath)

# 2 Compute geometry matrix
raw_rx1_geom_pd = steering_vectors.compute_geometry_matrix(raw_pd=raw_rx1_pd, ephemeris_filepath=ephemeris_filepath)
raw_rx2_geom_pd = steering_vectors.compute_geometry_matrix(raw_pd=raw_rx2_pd, ephemeris_filepath=ephemeris_filepath)

# 3 Compute pseudorange differences
raw_prd_pd = code_prd.get_prd(rx1_obs_pd=raw_rx1_geom_pd, rx2_obs_pd=raw_rx2_geom_pd, compute_dd=True)

# 4 Compute baseline
baseline_pd, raw_pd = code_prd.compute_baseline(rx_obs_pd=raw_prd_pd, weights_column="weight")
```
or more compact:
```python
import os
import bits

from bits_prd.src import code_prd

# 1 Parse data from two receivers with corresponding ephemeris
data_filepath = os.path.join(os.getcwd(), "bits_prd", "test", "data")
# Raw data
rx1_raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
rx2_raw_filepath = os.path.join(data_filepath, "RX0200FRA_R_20261241729_00U_01S_MO.rnx")
# Ephemeris
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

# Parse
raw_rx1_pd = bits.parsers.gnss_raw.rinex_obs(rx1_raw_filepath)
raw_rx2_pd = bits.parsers.gnss_raw.rinex_obs(rx2_raw_filepath)

# 2 Compute everything at once
baseline_pd, raw_pd = code_prd.compute_baseline(rx_obs_pd=raw_rx1_pd, rx2_obs_pd=raw_rx2_pd, weights_column="weight", 
                                                compute_dd=True, ephemeris_filepath=ephemeris_filepath)
```