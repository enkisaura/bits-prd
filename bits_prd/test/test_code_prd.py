#!/usr/bin/env python3

"""
Tests for code based PRD functions

Usage: Used from pytest
======
    python -m pytest -v
"""

import os
import bits

from bits_prd.src import code_prd

baseline_length = 20
uncertainty = 5

data_filepath = os.path.join(os.getcwd(), "bits_prd", "test", "data")
rx1_nmea_filepath = os.path.join(data_filepath, "RX01_nmea.txt")
rx2_nmea_filepath = os.path.join(data_filepath, "RX02_nmea.txt")
rx1_raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
rx2_raw_filepath = os.path.join(data_filepath, "RX0200FRA_R_20261241729_00U_01S_MO.rnx")
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

# Parse data
raw_rx1_pd = bits.parsers.gnss_raw.rinex_obs(rx1_raw_filepath)
raw_rx2_pd = bits.parsers.gnss_raw.rinex_obs(rx2_raw_filepath)
rx1_nmea_pd = bits.parsers.nmea.gga(rx1_nmea_filepath)
rx2_nmea_pd = bits.parsers.nmea.gga(rx2_nmea_filepath)

def test_sd():
    prd(compute_dd=False)

def test_dd():
    prd(compute_dd=True)

def prd(compute_dd:bool):
    baseline_pd, raw_pd = code_prd.compute_baseline(rx_obs_pd=raw_rx1_pd, rx2_obs_pd=raw_rx2_pd, compute_dd=compute_dd,
    ephemeris_filepath=ephemeris_filepath, pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    txt = f"Baseline estimate does not meet the expected accuracy. Expected: {baseline_length}+/-{uncertainty}m, estimated: mean {baseline_pd["baseline_m"].mean()}m, max {baseline_pd["baseline_m"].max()}m."
    assert abs(baseline_pd["baseline_m"].max() - baseline_length) < uncertainty, txt


if __name__ == '__main__':
    test_sd()
    test_dd()