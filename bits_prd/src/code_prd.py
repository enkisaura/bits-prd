import warnings
from typing import Literal, List
import numpy as np
import pandas as pd
from tqdm import tqdm

import bits # Available at https://github.com/enkisaura/Baguette-In-The-Sky.git
from bits_prd.src.steering_vectors import compute_geometry_matrix


def get_prd(rx1_obs_pd: pd.DataFrame, rx2_obs_pd: pd.DataFrame, time_between_meas:float=1, compute_dd=True,
            pivot_sv_id:str|None=None) -> pd.DataFrame:
    """
    Computes single and double differences. This functions merges rx1_obs_pd with rx2_obs_pd and adds pseudorange
    differences. In the returned dataframe, rx1_obs_pd data is noted "_rx1" and rx2_obs_pd "_rx2".

    :param rx1_obs_pd: BITS raw dataframe with geometry matrix
    :param rx2_obs_pd: BITS raw dataframe with geometry matrix
    :param time_between_meas: Maximum allowed time between measurements
    :param compute_dd: set to True to compute SD + DD, False for SD only
    :param pivot_sv_id: name of the pivot SV; set to None for no pivot
    :return: BITS raw dataframe like with sd and/or dd
    """
    # Compute single differences
    out_pd = get_single_difference(rx1_obs_pd, rx2_obs_pd, dt_tolerance=time_between_meas/2)

    # Compute double differences if applicable
    if compute_dd:
        out_pd = get_double_difference_no_pivot(out_pd)

        # Keep only DD with given pivot_sv_id if applicable
        if pivot_sv_id is not None:
            out_pd = out_pd[(out_pd["sv_id1"==pivot_sv_id]) | (out_pd["sv_id2"==pivot_sv_id])]

    return out_pd


def get_single_difference(rx1_obs_pd: pd.DataFrame, rx2_obs_pd: pd.DataFrame,
                          dt_tolerance: float = 0.5) -> pd.DataFrame:
    """
    Computes single difference. In the returned dataframe, rx1_obs_pd data is noted "_rx1" and rx2_obs_pd "_rx2".

    :param rx1_obs_pd: BITS raw dataframe
    :param rx2_obs_pd: BITS raw dataframe
    :param dt_tolerance: Maximum time between measurements of rx1 and rx2 to be considered at same timestamp
    :return: BITS raw dataframe like with sd
    """
    # 0 Clean up
    rx1_obs_pd = rx1_obs_pd.sort_values("time")
    rx2_obs_pd = rx2_obs_pd.sort_values("time")

    # 1 Merge common satellites from rx1_obs_pd and rx2_obs_pd
    rx2_obs_pd = rx2_obs_pd.assign(time_rx2=rx2_obs_pd["time"])
    out_pd = pd.merge_asof(
        rx1_obs_pd, rx2_obs_pd,
        on="time",
        by="sv_id",
        tolerance=pd.Timedelta(seconds=dt_tolerance),
        direction="nearest",
        suffixes=("_rx1", "_rx2")
    )

    # 2 Compute single differences
    out_pd["sd"] = out_pd["pr_m_rx1"] - out_pd["pr_m_rx2"]
    if ("pr_rate_mps_rx1" in out_pd.columns) and ("pr_rate_mps_rx2" in out_pd.columns):
        out_pd["sd_rate"] = out_pd["pr_rate_mps_rx1"] - out_pd["pr_rate_mps_rx2"]

    # Clean up
    out_pd.dropna(subset=["sd", "sd_rate"], inplace=True) # Drops non common satellites
    out_pd = out_pd.sort_values(by=["time", "sv_id"]).reset_index(drop=True)

    return out_pd


def get_double_difference_no_pivot(sd_obs_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Computes every possible double differences with no regards for pivot satellite. Requires sd measurements and
    steering vectors noted "_rx1" and "_rx2" depending on the receiver. Double differences combines measurements from
    two different satellites that will be noted "_sv1" and "_sv2".

    :param sd_obs_pd: BITS raw dataframe like with sd
    :return: BITS raw dataframe like with dd
    """
    if "sd_rate" in sd_obs_pd.columns:
        sd_rate = True
    else:
        sd_rate = False

    # Group by timestamp
    out_pd_list = []
    for _, group in tqdm(sd_obs_pd.groupby("time"), total=len(sd_obs_pd["time"].unique()),
                                     desc="Computing double differences"):
        at_timestamp_pd_list = []

        # Find every possible dd combination
        for i in range(group.shape[0] - 1):
            local_dd_pd = group[i:].copy()
            local_dd_pd["sv_id1"] = local_dd_pd["sv_id"].iloc[0]
            local_dd_pd["prn_id1"] = local_dd_pd["prn_id_rx1"].iloc[0]
            # Store data from local pivot SV as "_sv1"
            local_dd_pd["sd1"] = local_dd_pd["sd"].iloc[0]
            if sd_rate:
                local_dd_pd["sd_rate1"] = local_dd_pd["sd_rate"].iloc[0]
            local_dd_pd["e_x_sv1"] = local_dd_pd["e_x_rx1"].iloc[0]
            local_dd_pd["e_y_sv1"] = local_dd_pd["e_y_rx1"].iloc[0]
            if "e_z_rx1" in local_dd_pd.columns:
                two_d = False
                local_dd_pd["e_z_sv1"] = local_dd_pd["e_z_rx1"].iloc[0]
            else:
                two_d = True
            local_dd_pd = local_dd_pd[1:]

            at_timestamp_pd_list.append(local_dd_pd)
        try:
            at_timestamp_pd = pd.concat(at_timestamp_pd_list, ignore_index=True)
        except:
            continue

        out_pd_list.append(at_timestamp_pd)

    out_pd = pd.concat(out_pd_list, ignore_index=True)

    # Rename second SV as "_sv2"
    out_pd.rename(columns={"sv_id": "sv_id2", "prn_id": "prn_id2", "sd": "sd2",
                           "e_x_rx1": "e_x_sv2",
                           "e_y_rx1": "e_y_sv2"}, inplace=True)
    if sd_rate:
        out_pd.rename(columns={"sd_rate": "sd_rate2"}, inplace=True)
    if not two_d:
        out_pd.rename(columns={"e_z_rx1": "e_z_sv2"}, inplace=True)

    # Compute DD
    out_pd["dd"] = out_pd["sd1"] - out_pd["sd2"]
    if sd_rate:
        out_pd["dd_rate"] = out_pd["sd_rate1"] - out_pd["sd_rate2"]

    # Add differenced steering vectors
    out_pd["delta_e_x"] = (out_pd["e_x_sv1"] - out_pd["e_x_sv2"])
    out_pd["delta_e_y"] = (out_pd["e_y_sv1"] - out_pd["e_y_sv2"])
    if not two_d:
        out_pd["delta_e_z"] = (out_pd["e_z_sv1"] - out_pd["e_z_sv2"])

    # Clean up
    out_pd = out_pd.sort_values(by=["time", "sv_id1", "sv_id2"]).reset_index(drop=True)

    return out_pd

def decorrelate_space(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    For short baselines, the satellite radial velocities are strongly correlated between the two receivers and,
    therefore, cancel out when applying pseudorange rate differencing. However, as the baseline length increases, this
    correlation diminishes. As a result, the relative speed estimates derived from pseudorange rate differencing no
    longer accurately represent the true relative velocity between the receivers.

    :param obs_df: BITS raw dataframe with pseudorange rate, steering vectors and satellite speed
    :return: BITS raw dataframe with corrected "pr_rate_mps"
    """
    if not np.isin(["pr_rate_mps", "e_x", "e_y", "e_z", "vx_sv_mps", "vy_sv_mps", "vz_sv_mps"], obs_df.columns).all():
        warnings.warn("Need pseudorange rate, steering vectors and satellite speed to decorrelate space.")

    obs_df.rename(columns={"pr_rate_mps": "raw_pr_rate_mps"}, inplace=True)

    pr_rate = obs_df["raw_pr_rate_mps"].to_numpy()
    X_dot = np.vstack((obs_df["vx_sv_mps"], obs_df["vy_sv_mps"], obs_df["vz_sv_mps"]))
    G = np.vstack((obs_df["e_x"], obs_df["e_y"], obs_df["e_z"]))

    obs_df["pr_rate_mps"] = pr_rate + np.sum(X_dot * G, axis=0)

    return obs_df


def decorrelate_time(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    By considering two receivers A and B sharing their measurement at time tA and tB , applying PRD between those
    measurements would result in a baseline estimate between receiver A at time tA and B at time tB. With moving
    receivers, this baseline estimate does not represent effectively an actual distance between receivers anymore.

    b(t1/2) = b(tA, tB) + 1/2 (tA-tB) db

    :param baseline_df:
    :return:
    """
    if not np.isin(["vbx_rx_mps", "vby_rx_mps", "vbaseline_mps"], baseline_df.columns).all():
        warnings.warn("No baseline rate estimate found, cannot decorrelate time.")
    # Estimate receivers offset
    if "bb_rx_m" in baseline_df.columns:
        dt = np.asarray(baseline_df["bb_rx_m"] / bits.const.C, dtype=np.float64)
    elif "time_rx2" in baseline_df.columns:
        dt = np.asarray((baseline_df["time"] - baseline_df["time_rx2"]) / np.timedelta64(1, "s"), dtype=np.float64)
    else:
        warnings.warn("No between receivers time bias found, cannot decorrelate time.")
        return baseline_df

    baseline_df.rename(columns={"bx_rx_m": "raw_bx_rx_m", "by_rx_m": "raw_by_rx_m", "bb_rx_m": "raw_bb_rx_m",
                                "baseline_m": "raw_baseline_m", "time": "raw_time"}, inplace=True)

    offset = pd.to_timedelta(np.nan_to_num(0.5 * dt, nan=0.0), unit="s")
    baseline_df["time"] = baseline_df["raw_time"] + offset
    baseline_df["bx_rx_m"] = baseline_df["raw_bx_rx_m"] + 0.5 * dt * baseline_df["vbx_rx_mps"]
    baseline_df["by_rx_m"] = baseline_df["raw_by_rx_m"] + 0.5 * dt * baseline_df["vby_rx_mps"]
    if "bz_rx_m" in baseline_df.columns:
        baseline_df.rename(columns={"bz_rx_m": "raw_bz_rx_m"}, inplace=True)
        baseline_df["bz_rx_m"] = baseline_df["raw_bz_rx_m"] + 0.5 * dt * baseline_df["vbz_rx_mps"]
    baseline_df["raw_bb_rx_m"] = 0
    baseline_df["baseline_m"] = baseline_df["raw_baseline_m"] + 0.5 * dt * baseline_df["vbaseline_mps"]

    return baseline_df


def compute_baseline(rx_obs_pd: pd.DataFrame, rx2_obs_pd: None|pd.DataFrame = None, weights_column:str="weight",
                     time_between_meas:float = 1, compute_dd:None|bool=None, horizontal:bool=False, pivot_sv_id:str | None = None,
                     ephemeris_pd: pd.DataFrame | None = None, ephemeris_filepath: str | None = None,
                     pos_pd_rx1:pd.DataFrame | None = None, pos_pd_rx2:pd.DataFrame | None = None,
                     corrections:List[Literal["space_decorrelation", "time_decorrelation", "all"]] = ["all"]) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute baseline with SD or DD. This function is not able to compute SD and DD at the same time.

    :param rx_obs_pd: BITS raw dataframe with SD/DD computed or associated with rx2_obs_pd
    :param rx2_obs_pd: BITS raw dataframe
    :param weights_column: Name of the column containing the weights if any
    :param time_between_meas: Maximum allowed time between measurements
    :param compute_dd: set to True to compute SD + DD, False for SD only
    :param pivot_sv_id: name of the pivot SV; set to None for no pivot
    :param ephemeris_pd: BITS ephemeris dataframe
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :param pos_pd_rx1: BITS PVT dataframe associated with rx_obs_pd
    :param pos_pd_rx2: BITS PVT dataframe associated with rx2_obs_pd
    :param corrections: Corrections to be applied ("space_decorrelation", "time_decorrelation", "all"), defaults to all
    :return: BITS PVT like dataframe with baseline estimate, BITS raw like dataframe
    """
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
    if "e_x" not in rx_obs_pd.columns or "e_x_rx1" not in rx_obs_pd.columns:
        pos_pd_rx1, rx_obs_pd = compute_geometry_matrix(rx_obs_pd, ephemeris_pd=ephemeris_pd,
                                            ephemeris_filepath=ephemeris_filepath, pos_pd=pos_pd_rx1)

    if rx2_obs_pd is not None:
        if "e_x" not in rx2_obs_pd.columns:
            pos_pd_rx2, rx2_obs_pd = compute_geometry_matrix(rx2_obs_pd, ephemeris_pd=ephemeris_pd,
                                                 ephemeris_filepath=ephemeris_filepath, pos_pd=pos_pd_rx2)

    # 1 bis space decorrelation
    if ("space_decorrelation" in corrections) or ("all" in corrections):
        rx_obs_pd = decorrelate_space(rx_obs_pd)
        rx2_obs_pd = decorrelate_space(rx2_obs_pd)

    # 1 ter convert geometry matrix to ENU for horizontal baseline
    if horizontal:
        approx_pos = pos_pd_rx1[["x_rx_m", "y_rx_m", "z_rx_m"]].dropna().iloc[0]
        enu_rx1 = bits.convert.space.ecef_to_enu(rx_obs_pd["e_x"], rx_obs_pd["e_y"], rx_obs_pd["e_z"],
                                                 approx_pos["x_rx_m"], approx_pos["y_rx_m"], approx_pos["z_rx_m"],
                                                 with_translation=False)

        rx_obs_pd["e_x"] = enu_rx1[0]
        rx_obs_pd["e_y"] = enu_rx1[1]
        rx_obs_pd.drop(["e_z"], axis=1, inplace=True)

        if rx2_obs_pd is not None:
            enu_rx2 = bits.convert.space.ecef_to_enu(rx2_obs_pd["e_x"], rx2_obs_pd["e_y"], rx2_obs_pd["e_z"],
                                                     approx_pos["x_rx_m"], approx_pos["y_rx_m"], approx_pos["z_rx_m"],
                                                     with_translation=False)
            rx2_obs_pd["e_x"] = enu_rx2[0]
            rx2_obs_pd["e_y"] = enu_rx2[1]
            rx2_obs_pd.drop(["e_z"], axis=1, inplace=True)

    # 2 Compute Single/Double differences
    if rx2_obs_pd is not None:
        rx_obs_pd = get_prd(rx_obs_pd, rx2_obs_pd, time_between_meas=time_between_meas, compute_dd=compute_dd,
        pivot_sv_id=pivot_sv_id).dropna()

    # 3 Compute baseline
    # Group by timestamp
    tqdm_desc = f"Computing baseline with {mode}"
    raw_pd_list = []
    baseline_pd_list = []
    for _, group in tqdm(rx_obs_pd.groupby("time"), total=len(rx_obs_pd["time"].unique()), desc=tqdm_desc):
        baseline_pd, raw_pd = window_compute_baseline(group, mode=mode, weights_column=weights_column)
        raw_pd_list.append(raw_pd)
        baseline_pd_list.append(baseline_pd)

    # Merge all timestamps
    raw_pd = pd.concat(raw_pd_list, ignore_index=True)
    baseline_pd = pd.concat(baseline_pd_list, ignore_index=True)

    # Clean
    raw_pd["time"] = raw_pd["time"].astype("datetime64[ns]")
    baseline_pd["time"] = baseline_pd["time"].astype("datetime64[ns]")
    if "time_rx2" in raw_pd.columns:
        raw_pd["time_rx2"] = raw_pd["time_rx2"].astype("datetime64[ns]")
    if "time_rx2" in baseline_pd.columns:
        baseline_pd["time_rx2"] = baseline_pd["time_rx2"].astype("datetime64[ns]")

    # 3 bis time decorrelation
    if ("time_decorrelation" in corrections) or ("all" in corrections):
        baseline_pd = decorrelate_time(baseline_pd)

    return baseline_pd, raw_pd

def window_compute_baseline(group: pd.DataFrame, mode:Literal["sd", "dd"]="dd", weights_column:str="weight") \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute baseline at a specific timestamp.

    :param group: BITS raw dataframe with geometry matrix and SD/DD computed at a single epoch
    :param mode: set to "sd" or "dd" depending on data in group
    :param weights_column: Name of the column containing the weights if any
    :return: BITS raw like dataframe, BITS PVT like Serie with baseline estimate
    """
    two_d = False
    # Build measurement matrix
    Y = group[mode].to_numpy().reshape(-1, 1)

    # Build geometry matrix
    if mode == "sd":
        ex = group["e_x_rx1"].to_numpy()
        ey = group["e_y_rx1"].to_numpy()
        if "e_z_rx1" in group:
            ez = group["e_z_rx1"].to_numpy()
            G = np.vstack((ex, ey, ez, np.ones_like(ex))).transpose() # Add a ones column for inter-rx clock bias
        else:
            two_d = True
            G = np.vstack((ex, ey, np.ones_like(ex))).transpose()  # Add a ones column for inter-rx clock bias
    else:
        ex = group["delta_e_x"].to_numpy()
        ey = group["delta_e_y"].to_numpy()
        if "delta_e_z" in group:
            ez = group["delta_e_z"].to_numpy()
            G = np.vstack((ex, ey, ez)).transpose()
        else:
            two_d = True
            G = np.vstack((ex, ey)).transpose()

    # Build weight matrix
    if weights_column in group.columns:
        w = group[weights_column].to_numpy()
    else:
        w = np.ones_like(Y)
    W = np.diag(w.ravel())

    # Compute baseline
    try:
        result = bits.single_point_positioning.weighted_least_square(Y, G, W)
    except:
        result = None

    # Save the result
    baseline_serie = pd.Series({"time": group["time"].iloc[0],
                                "mode": mode,})
    if "time_rx2" in group.columns:
        baseline_serie["time_rx2"] = group["time_rx2"].iloc[0]

    if result is not None:
        estimate, covariance, dop, residuals = result

        group["residuals_m"] = residuals

        baseline_serie["bx_rx_m"] = float(estimate[0][0])
        baseline_serie["by_rx_m"] = float(estimate[1][0])
        if not two_d:
            baseline_serie["bz_rx_m"] = float(estimate[2][0])
        if mode == "sd":
            baseline_serie["bb_rx_m"] = float(estimate[-1][0])
        if two_d:
            baseline_serie["baseline_m"] = float(np.linalg.norm(estimate[:2]))
        else:
            baseline_serie["baseline_m"] = float(np.linalg.norm(estimate[:3]))

        baseline_serie["cov_xx_rx_m"] = float(covariance[0][0])
        baseline_serie["cov_yx_rx_m"] = float(covariance[0][1])
        baseline_serie["cov_yy_rx_m"] = float(covariance[1][1])
        if not two_d:
            baseline_serie["cov_zx_rx_m"] = float(covariance[0][2])
            baseline_serie["cov_zy_rx_m"] = float(covariance[1][2])
            baseline_serie["cov_zz_rx_m"] = float(covariance[2][2])
        if mode == "sd":
            baseline_serie["cov_bx_rx_m"] = float(covariance[0][-1])
            baseline_serie["cov_by_rx_m"] = float(covariance[1][-1])
            if not two_d:
                baseline_serie["cov_bz_rx_m"] = float(covariance[2][3])
            baseline_serie["cov_bb_rx_m"] = float(covariance[-1][-1])

        baseline_serie["DOP"] = float(dop)
    else:
        group["residuals_m"] = None

        baseline_serie["bx_rx_m"] = None
        baseline_serie["by_rx_m"] = None
        if not two_d:
            baseline_serie["bz_rx_m"] = None
        if mode == "sd":
            baseline_serie["bb_rx_m"] = None
        baseline_serie["baseline_m"] = None

        baseline_serie["cov_xx_rx_m"] = None
        baseline_serie["cov_yx_rx_m"] = None
        baseline_serie["cov_yy_rx_m"] = None
        if not two_d:
            baseline_serie["cov_zx_rx_m"] = None
            baseline_serie["cov_zy_rx_m"] = None
            baseline_serie["cov_zz_rx_m"] = None
        if mode == "sd":
            baseline_serie["cov_bx_rx_m"] = None
            baseline_serie["cov_by_rx_m"] = None
            if not two_d:
                baseline_serie["cov_bz_rx_m"] = None
            baseline_serie["cov_bb_rx_m"] = None

        baseline_serie["DOP"] = None

    # Compute speed
    if f"{mode}_rate" in group.columns:
        dY = group[f"{mode}_rate"].to_numpy().reshape(-1, 1)
        try:
            speed_result = bits.single_point_positioning.weighted_least_square(dY, G, W)
        except:
            speed_result = None

        # Save the result
        if speed_result is not None:
            v_estimate, v_covariance, _, v_residuals = speed_result

            group["vresiduals_mps"] = residuals

            baseline_serie["vbx_rx_mps"] = float(v_estimate[0][0])
            baseline_serie["vby_rx_mps"] = float(v_estimate[1][0])
            if not two_d:
                baseline_serie["vbz_rx_mps"] = float(v_estimate[2][0])
            if mode == "sd":
                baseline_serie["vbb_rx_mps"] = float(v_estimate[-1][0])
            if two_d:
                baseline_serie["vbaseline_mps"] = float(np.linalg.norm(v_estimate[:2]))
            else:
                baseline_serie["vbaseline_mps"] = float(np.linalg.norm(v_estimate[:3]))

            baseline_serie["cov_vxvx_rx_mps"] = float(v_covariance[0][0])
            baseline_serie["cov_vyvx_rx_mps"] = float(v_covariance[0][1])
            baseline_serie["cov_vyvy_rx_mps"] = float(v_covariance[1][1])
            if not two_d:
                baseline_serie["cov_vzvx_rx_mps"] = float(v_covariance[0][2])
                baseline_serie["cov_vzvy_rx_mps"] = float(v_covariance[1][2])
                baseline_serie["cov_vzvz_rx_mps"] = float(v_covariance[2][2])
            if mode == "sd":
                baseline_serie["cov_vbvx_rx_mps"] = float(v_covariance[0][-1])
                baseline_serie["cov_vbvy_rx_mps"] = float(v_covariance[1][-1])
                if not two_d:
                    baseline_serie["cov_vbvz_rx_mps"] = float(v_covariance[2][-1])
                baseline_serie["cov_vbvb_rx_mps"] = float(v_covariance[-1][-1])

        else:
            group["vresiduals_mps"] = None

            baseline_serie["vbx_rx_mps"] = None
            baseline_serie["vby_rx_mps"] = None
            if two_d:
                baseline_serie["vbz_rx_mps"] = None
            if mode == "sd":
                baseline_serie["vbb_rx_mps"] = None
            baseline_serie["vbaseline_mps"] = None

            baseline_serie["cov_vxvx_rx_mps"] = None
            baseline_serie["cov_vyvx_rx_mps"] = None
            baseline_serie["cov_vyvy_rx_mps"] = None
            if not two_d:
                baseline_serie["cov_vzvx_rx_mps"] = None
                baseline_serie["cov_vzvy_rx_mps"] = None
                baseline_serie["cov_vzvz_rx_mps"] = None
            if mode == "sd":
                baseline_serie["cov_vbvx_rx_mps"] = None
                baseline_serie["cov_vbvy_rx_mps"] = None
                if not two_d:
                    baseline_serie["cov_vbvz_rx_mps"] = None
                baseline_serie["cov_vbvb_rx_mps"] = None


    return baseline_serie.to_frame().T, group
