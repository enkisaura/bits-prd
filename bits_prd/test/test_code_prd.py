#!/usr/bin/env python3

"""
Tests for code based PRD functions

Usage: Used from pytest
======
    python -m pytest -v
"""

import pandas as pd
import numpy as np
from bits import parse
from bits.src.utils import get_example_data_filepath, fast_parse

from bits_prd.src import code_prd

uncertainty = 12
speed_uncertainty = 1

# Parse data
raw_rx1_pd = fast_parse(get_example_data_filepath("raw", rover_type="fixed")[0], parse.raw.rinex)
raw_rx2_pd = fast_parse(get_example_data_filepath("raw", rover_type="circular")[0], parse.raw.rinex)
rx1_nmea_pd = fast_parse(get_example_data_filepath("pvt", rover_type="fixed")[0], parse.pvt.rmc)
rx2_nmea_pd = fast_parse(get_example_data_filepath("pvt", rover_type="circular")[0], parse.pvt.rmc)
ephemeris_df = fast_parse(get_example_data_filepath("ephemeris", rover_type="fixed")[0], parse.ephemeris.rinex)

# Get ground truth
gt_pd = pd.merge_asof(
    rx1_nmea_pd, rx2_nmea_pd,
    on="time",
    tolerance=pd.Timedelta(seconds=0.5),
    direction="nearest",
    suffixes=("_rx1", "_rx2")
)
gt_pd["baseline_m"] = np.sqrt((gt_pd["x_rx_m_rx1"] - gt_pd["x_rx_m_rx2"]) ** 2 +
                              (gt_pd["y_rx_m_rx1"] - gt_pd["y_rx_m_rx2"]) ** 2 +
                              (gt_pd["z_rx_m_rx1"] - gt_pd["z_rx_m_rx2"]) ** 2)
gt_pd["vbaseline_mps"] = np.sqrt((gt_pd["vx_rx_mps_rx1"] - gt_pd["vx_rx_mps_rx2"]) ** 2 +
                                 (gt_pd["vy_rx_mps_rx1"] - gt_pd["vy_rx_mps_rx2"]) ** 2 +
                                 (gt_pd["vz_rx_mps_rx1"] - gt_pd["vz_rx_mps_rx2"]) ** 2)
gt_pd = gt_pd[["time", "baseline_m", "vbaseline_mps"]]


def test_sd(verbose=False):
    prd(compute_dd=False, verbose=verbose)

def test_dd(verbose=False):
    prd(compute_dd=True, verbose=verbose)

def prd(compute_dd:bool, verbose=False):
    baseline_pd, raw_pd = code_prd.compute_baseline(rx_obs_pd=raw_rx1_pd, rx2_obs_pd=raw_rx2_pd, compute_dd=compute_dd,
    ephemeris_pd=ephemeris_df, pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # Get error
    baseline_pd = pd.merge_asof(
        baseline_pd, gt_pd,
        on="time",
        tolerance=pd.Timedelta(seconds=0.5),
        direction="nearest",
        suffixes=("", "_gt")
    )
    baseline_pd["baseline_error_m"] = np.abs(baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"])
    baseline_pd["vbaseline_error_mps"] = np.abs(baseline_pd["vbaseline_mps"] - baseline_pd["vbaseline_mps_gt"])

    report = (f"Expected position accuracy: {uncertainty}m, estimated: mean {baseline_pd["baseline_error_m"].mean()}m, max {baseline_pd["baseline_error_m"].max()}m.\n"
              f"Expected speed accuracy: {speed_uncertainty}m/s, estimated: mean {baseline_pd["vbaseline_error_mps"].mean()}m/s, max {baseline_pd["vbaseline_error_mps"].max()}m/s.")
    if verbose:
        print(report)

    txt = f"Baseline estimate does not meet the expected accuracy. {report}"
    assert baseline_pd["baseline_error_m"].max() < uncertainty, txt
    assert baseline_pd["vbaseline_error_mps"].max() < speed_uncertainty, txt


if __name__ == '__main__':
    test_sd(verbose=True)
    test_dd(verbose=True)