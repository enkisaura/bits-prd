import bits
from bits_prd.src import utils

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_prd(rx1_obs_pd: pd.DataFrame, rx2_obs_pd: pd.DataFrame, dt_tolerance=0.5, compute_dd=True,
            pivot_full_sv_id:str|None=None) -> pd.DataFrame:
    """
    Computes single and double differences
    :param rx1_obs_pd: need at least pr_m, steering vectors, unix_time, full_sv_id
    :param rx2_obs_pd: need at least pr_m, steering vectors, unix_time, full_sv_id
    :param dt_tolerance:
    :param compute_dd: set to True to compute SD + DD, False for SD only
    :param pivot_full_sv_id: name of the pivot SV
    :return:
    """
    out_pd = get_single_difference(rx1_obs_pd, rx2_obs_pd, dt_tolerance=dt_tolerance)

    if compute_dd:
        out_pd = get_double_difference_no_pivot(out_pd)

        if pivot_full_sv_id is not None:
            out_pd = out_pd[(out_pd["full_sv_id1"==pivot_full_sv_id]) | (out_pd["full_sv_id2"==pivot_full_sv_id])] #TODO

    return out_pd


def get_single_difference(rx1_obs_pd: pd.DataFrame, rx2_obs_pd: pd.DataFrame,
                          dt_tolerance: float = 0.5) -> pd.DataFrame:
    # 1 At each timestamp, merge common sattellites
    rx1_obs_pd = rx1_obs_pd.sort_values("unix_time")
    rx2_obs_pd = rx2_obs_pd.sort_values("unix_time")

    out_pd = pd.merge_asof(
        rx1_obs_pd, rx2_obs_pd,
        on="unix_time",
        by="full_sv_id",
        tolerance=dt_tolerance,
        direction="nearest",
        suffixes=("_rx1", "_rx2")
    )

    # 2 Compute single differences
    out_pd["sd"] = out_pd["pr_m_rx1"] - out_pd["pr_m_rx2"]

    # Clean up
    out_pd.dropna(subset=["sd"], inplace=True)
    out_pd = out_pd.sort_values(by=["unix_time", "full_sv_id"]).reset_index(drop=True)

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
            local_dd_pd["full_sv_id1"] = local_dd_pd["full_sv_id"].iloc[0]
            local_dd_pd["sv_id1"] = local_dd_pd["sv_id_rx1"].iloc[0]
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
    out_pd.rename(columns={"full_sv_id": "full_sv_id2", "sv_id": "sv_id2", "sd": "sd2",
                           "steering_vector_x_rx1": "steering_vector_x_sv2",
                           "steering_vector_y_rx1": "steering_vector_y_sv2",
                           "steering_vector_z_rx1": "steering_vector_z_sv2"}, inplace=True)

    # Compute DD
    out_pd["dd"] = out_pd["sd1"] - out_pd["sd2"]

    # Add differenced steering vectors
    out_pd["delta_steering_vector_x"] = (out_pd["steering_vector_x_sv1"] - out_pd["steering_vector_x_sv2"])
    out_pd["delta_steering_vector_y"] = (out_pd["steering_vector_y_sv1"] - out_pd["steering_vector_y_sv2"])
    out_pd["delta_steering_vector_z"] = (out_pd["steering_vector_z_sv1"] - out_pd["steering_vector_z_sv2"])

    out_pd = out_pd.sort_values(by=["unix_time", "full_sv_id1", "full_sv_id2"]).reset_index(drop=True)  # Clean up

    return out_pd


def compute_baseline(prd_pd: pd.DataFrame, weights_column:str="weight") -> pd.DataFrame:
    if "dd" in prd_pd.columns:
        mode = "dd"
    elif "sd" in prd_pd.columns:
        mode = "sd"
    else:
        raise ValueError("Neither 'dd' or 'sd' columns present, cannot compute baseline with code based pseudorange "
                         "differencing.")

    # Group by timestamp
    tqdm_desc = f"Computing baseline with {mode}"

    out_pd_list = []
    for _, group in tqdm(prd_pd.groupby("unix_time"), total=len(prd_pd["unix_time"].unique()), desc=tqdm_desc):
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
        if weights_column in prd_pd.columns:
            w = group[weights_column].to_numpy()
        else:
            w = np.ones_like(Y)
        W = np.diag(w.ravel())

        # Compute baseline
        result = utils.wls(Y, G, W)

        # Save the result
        out_group = group.copy()
        if result is not None:
            estimate, covariance = result
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

        out_pd_list.append(out_group)

    return pd.concat(out_pd_list, ignore_index=True)
