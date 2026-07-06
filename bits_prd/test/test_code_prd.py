#!/usr/bin/env python3

"""
Tests for code based PRD functions

Usage: Used from pytest
======
    python -m pytest -v
"""

import os
import bits

from bits_prd.src import steering_vectors, code_prd

baseline_length = 20
uncertainty = 5

data_filepath = os.path.join(os.getcwd(), "bits_prd", "test", "data")
rx1_nmea_filepath = os.path.join(data_filepath, "RX01_nmea.txt")
rx2_nmea_filepath = os.path.join(data_filepath, "RX02_nmea.txt")
rx1_raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
rx2_raw_filepath = os.path.join(data_filepath, "RX0200FRA_R_20261241729_00U_01S_MO.rnx")
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

# Parse data
ephemeris_pd = bits.parsers.ephemeris.rinex_nav(ephemeris_filepath)
raw_rx1_pd = bits.parsers.gnss_raw.rinex_obs(rx1_raw_filepath)
raw_rx2_pd = bits.parsers.gnss_raw.rinex_obs(rx2_raw_filepath)
rx1_nmea_pd = bits.parsers.nmea.gga(rx1_nmea_filepath)
rx2_nmea_pd = bits.parsers.nmea.gga(rx2_nmea_filepath)

# Get geometry matrix
gnss_data_rx1_pd = steering_vectors.compute_geometry_matrix(raw_rx1_pd, pos_pd=rx1_nmea_pd, ephemeris_pd=ephemeris_pd)
gnss_data_rx2_pd = steering_vectors.compute_geometry_matrix(raw_rx2_pd, pos_pd=rx2_nmea_pd, ephemeris_pd=ephemeris_pd)

def test_sd():
    psrd_pd = code_prd.get_prd(gnss_data_rx1_pd, gnss_data_rx2_pd, compute_dd=False, dt_tolerance=0.5)
    psrd_pd = code_prd.compute_baseline(psrd_pd)

    txt = f"Baseline estimate does not meet the expected accuracy. Expected: {baseline_length}+/-{uncertainty}m, estimated: mean {psrd_pd["baseline"].mean()}m, max {psrd_pd["baseline"].max()}m."
    assert abs(psrd_pd["baseline"].max() - baseline_length) < uncertainty, txt

def test_dd():
    psrd_pd = code_prd.get_prd(gnss_data_rx1_pd, gnss_data_rx2_pd, compute_dd=True, dt_tolerance=0.5)
    psrd_pd = code_prd.compute_baseline(psrd_pd)

    txt = f"Baseline estimate does not meet the expected accuracy. Expected: {baseline_length}+/-{uncertainty}m, estimated: mean {psrd_pd["baseline"].mean()}m, max {psrd_pd["baseline"].max()}m."
    assert abs(psrd_pd["baseline"].max() - baseline_length) < uncertainty, txt


if __name__ == '__main__':
    test_sd()
    test_dd()