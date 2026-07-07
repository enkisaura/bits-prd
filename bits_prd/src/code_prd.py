from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

import bits # Available at https://github.com/enkisaura/Baguette-In-The-Sky.git
from bits_prd.src.steering_vectors import compute_geometry_matrix


def get_prd(rx1_obs_pd: pd.DataFrame, rx2_obs_pd: pd.DataFrame, time_between_meas:float=1, compute_dd=True,
            pivot_sv_id:str|None=None) -> pd.DataFrame:
    """
    Computes single and double differences
    :param rx1_obs_pd: need at least pr_m, steering vectors, unix_time, sv_id
    :param rx2_obs_pd: need at least pr_m, steering vectors, unix_time, sv_id
    :param time_between_meas:
    :param compute_dd: set to True to compute SD + DD, False for SD only
    :param pivot_sv_id: name of the pivot SV
    :return:
    """
    if "unix_time" not in rx1_obs_pd.columns:
        rx1_obs_pd["unix_time"] = rx1_obs_pd["time"].apply(lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())
    out_pd = get_single_difference(rx1_obs_pd, rx2_obs_pd, dt_tolerance=time_between_meas/2)

    if compute_dd:
        out_pd = get_double_difference_no_pivot(out_pd)

        if pivot_sv_id is not None:
            out_pd = out_pd[(out_pd["sv_id1"==pivot_sv_id]) | (out_pd["sv_id2"==pivot_sv_id])] #TODO

    return out_pd


def get_single_difference(rx1_obs_pd: pd.DataFrame, rx2_obs_pd: pd.DataFrame,
                          dt_tolerance: float = 0.5) -> pd.DataFrame:
    # 1 At each timestamp, merge common sattellites
    rx1_obs_pd = rx1_obs_pd.sort_values("unix_time")
    rx2_obs_pd = rx2_obs_pd.sort_values("unix_time")

    out_pd = pd.merge_asof(
        rx1_obs_pd, rx2_obs_pd,
        on="unix_time",
        by="sv_id",
        tolerance=dt_tolerance,
        direction="nearest",
        suffixes=("_rx1", "_rx2")
    )

    # 2 Compute single differences
    out_pd["sd"] = out_pd["pr_m_rx1"] - out_pd["pr_m_rx2"]

    # Clean up
    out_pd.dropna(subset=["sd"], inplace=True)
    out_pd = out_pd.sort_values(by=["unix_time", "sv_id"]).reset_index(drop=True)

    return out_pd


def get_double_difference_no_pivot(sd_obs_pd: pd.DataFrame) -> pd.DataFrame:
    # Group by timestamp
    out_pd_list = []
    for _, group in tqdm(sd_obs_pd.groupby("unix_time"), total=len(sd_obs_pd["unix_time"].unique()),
                                     desc="Computing double differences"):
        at_timestamp_pd_list = []

        # Find every possible dd combination
        for i in range(group.shape[0] - 1):
            local_dd_pd = group[i:].copy()
            local_dd_pd["sv_id1"] = local_dd_pd["sv_id"].iloc[0]
            local_dd_pd["prn_id1"] = local_dd_pd["prn_id_rx1"].iloc[0]
            local_dd_pd["sd1"] = local_dd_pd["sd"].iloc[0]
            local_dd_pd["steering_vector_x_sv1"] = local_dd_pd["steering_vector_x_rx1"].iloc[0]
            local_dd_pd["steering_vector_y_sv1"] = local_dd_pd["steering_vector_y_rx1"].iloc[0]
            local_dd_pd["steering_vector_z_sv1"] = local_dd_pd["steering_vector_z_rx1"].iloc[0]
            local_dd_pd = local_dd_pd[1:]

            at_timestamp_pd_list.append(local_dd_pd)
        try:
            at_timestamp_pd = pd.concat(at_timestamp_pd_list, ignore_index=True)
        except:
            continue

        out_pd_list.append(at_timestamp_pd)

    out_pd = pd.concat(out_pd_list, ignore_index=True)

    # Rename sv2 data
    out_pd.rename(columns={"sv_id": "sv_id2", "prn_id": "prn_id2", "sd": "sd2",
                           "steering_vector_x_rx1": "steering_vector_x_sv2",
                           "steering_vector_y_rx1": "steering_vector_y_sv2",
                           "steering_vector_z_rx1": "steering_vector_z_sv2"}, inplace=True)

    # Compute DD
    out_pd["dd"] = out_pd["sd1"] - out_pd["sd2"]

    # Add differenced steering vectors
    out_pd["delta_steering_vector_x"] = (out_pd["steering_vector_x_sv1"] - out_pd["steering_vector_x_sv2"])
    out_pd["delta_steering_vector_y"] = (out_pd["steering_vector_y_sv1"] - out_pd["steering_vector_y_sv2"])
    out_pd["delta_steering_vector_z"] = (out_pd["steering_vector_z_sv1"] - out_pd["steering_vector_z_sv2"])

    out_pd = out_pd.sort_values(by=["unix_time", "sv_id1", "sv_id2"]).reset_index(drop=True)  # Clean up

    return out_pd


def compute_baseline(rx_obs_pd: pd.DataFrame, rx2_obs_pd: None|pd.DataFrame = None, weights_column:str="weight",
                     time_between_meas:float = 1, compute_dd:None|bool=None, pivot_sv_id:str | None = None,
                     ephemeris_pd: pd.DataFrame | None = None, ephemeris_filepath: str | None = None,
                     pos_pd_rx1:pd.DataFrame | None = None, pos_pd_rx2:pd.DataFrame | None = None) -> pd.DataFrame:
    # 0 determine mode (SD/DD); Default = DD
    if "dd" in rx_obs_pd.columns:
        mode = "dd"
        compute_dd = True
    elif "sd" in rx_obs_pd.columns:
        mode = "sd"
        compute_dd = False
    elif compute_dd is False:
        mode = "sd"
    else:
        mode = "dd"
        compute_dd = True

    if rx2_obs_pd is None and ("sd" not in rx_obs_pd.columns or "dd" not in rx_obs_pd.columns):
        raise ValueError("No pseudorange differences found in Dataframe. Please provide pseudoranges from a second "
                         "receiver in argument rx2_obs_pd.")

    # 1 Compute steering vectors
    if "steering_vector_x" not in rx_obs_pd.columns or "steering_vector_x_rx1" not in rx_obs_pd.columns:
        rx_obs_pd = compute_geometry_matrix(rx_obs_pd, ephemeris_pd=ephemeris_pd,
                                            ephemeris_filepath=ephemeris_filepath, pos_pd=pos_pd_rx1)

    if rx2_obs_pd is not None:
        if "steering_vector_x" not in rx2_obs_pd.columns:
            rx2_obs_pd = compute_geometry_matrix(rx2_obs_pd, ephemeris_pd=ephemeris_pd,
                                                 ephemeris_filepath=ephemeris_filepath, pos_pd=pos_pd_rx2)

    # 2 Compute Single/Double differences
    if rx2_obs_pd is not None:
        rx_obs_pd = get_prd(rx_obs_pd, rx2_obs_pd, time_between_meas=time_between_meas, compute_dd=compute_dd,
        pivot_sv_id=pivot_sv_id)

    # 3 Compute baseline
    # Group by timestamp
    tqdm_desc = f"Computing baseline with {mode}"

    out_pd_list = []
    for _, group in tqdm(rx_obs_pd.groupby("unix_time"), total=len(rx_obs_pd["unix_time"].unique()), desc=tqdm_desc):
        result_pd = window_compute_baseline(group, mode=mode, weights_column=weights_column)
        out_pd_list.append(result_pd)

    return pd.concat(out_pd_list, ignore_index=True)

def window_compute_baseline(group: pd.DataFrame, mode:Literal["sd", "dd"]="dd", weights_column:str="weight") -> pd.DataFrame:
    # Build measurement matrix
    Y = group[mode].to_numpy().reshape(-1, 1)

    # Build geometry matrix
    if mode == "sd":
        ex = group["steering_vector_x_rx1"].to_numpy()
        ey = group["steering_vector_y_rx1"].to_numpy()
        ez = group["steering_vector_z_rx1"].to_numpy()

        G = np.vstack((ex, ey, ez, np.ones_like(ex))).transpose()
    else:
        ex = group["delta_steering_vector_x"].to_numpy()
        ey = group["delta_steering_vector_y"].to_numpy()
        ez = group["delta_steering_vector_z"].to_numpy()

        G = np.vstack((ex, ey, ez)).transpose()

    # Build weight matrix
    if weights_column in group.columns:
        w = group[weights_column].to_numpy()
    else:
        w = np.ones_like(Y)
    W = np.diag(w.ravel())

    # Compute baseline
    try:
        result = bits.spp.weighted_least_square(Y, G, W)
    except:
        result = None

    # Save the result
    out_group = group.copy()
    if result is not None:
        estimate, covariance, _ = result
        residuals = Y - G @ estimate

        out_group["baseline_x"] = float(estimate[0][0])
        out_group["baseline_y"] = float(estimate[1][0])
        out_group["baseline_z"] = float(estimate[2][0])
        if mode == "sd":
            out_group["baseline_b"] = float(estimate[3][0])
        out_group["baseline"] = float(np.linalg.norm(estimate[:3]))
        out_group["covariance_x"] = float(covariance[0][0])
        out_group["covariance_y"] = float(covariance[1][1])
        out_group["covariance_z"] = float(covariance[2][2])
        if mode == "sd":
            out_group["covariance_b"] = float(covariance[3][3])
        out_group["uncertainty"] = float(np.sqrt(np.trace(covariance[:3, :3])))

        out_group["residuals"] = residuals
        out_group["std"] = float(np.std(residuals.ravel()))
    else:
        out_group["baseline_x"] = None
        out_group["baseline_y"] = None
        out_group["baseline_z"] = None
        if mode == "sd":
            out_group["baseline_b"] = None
        out_group["baseline"] = None
        out_group["covariance_x"] = None
        out_group["covariance_y"] = None
        out_group["covariance_z"] = None
        if mode == "sd":
            out_group["covariance_b"] = None

        out_group["residuals"] = None
        out_group["std"] = None

    return out_group
