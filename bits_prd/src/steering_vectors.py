import bits

import numpy as np
import pandas as pd

def compute_geometry_matrix(raw_pd: pd.DataFrame,
                            ephemeris_pd: pd.DataFrame|None=None, ephemeris_filepath: pd.DataFrame|None=None,
                            pos_pd:pd.DataFrame|None=None) -> pd.DataFrame:
    """
    Need either ephemeris_pd or ephemeris_filepath

    "unix_time": secondes
    "sv_id"
    "gnss_id"
    "x_rx_m", "y_rx_m", "z_rx_m": receiver pos (ECEF meters)

    :param raw_pd: raw measurements containing at least "unix_time", "sv_id" and "gnss_id"
    :param ephemeris_pd: Dataframe containing ephemeris. Use BITS parser
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :param pos_pd: Position of receiver containing at least "unix_time", "x_rx_m", "y_rx_m", "z_rx_m"
    :return:
    """
    if pos_pd is None:
        out_pd = slow_sv_pos(raw_pd, ephemeris_pd, ephemeris_filepath) # Also computes approximate position
    else:
        out_pd = fast_sv_pos(raw_pd, pos_pd, ephemeris_pd, ephemeris_filepath)

    # Compute steering vectors
    out_pd = compute_steering_vector(out_pd)

    # Add a unique SV identifier
    out_pd["full_sv_id"] = out_pd.apply(lambda row: f"{row["gnss_id"]}{row["sv_id"]}", axis=1)

    return out_pd


def slow_sv_pos(raw_pd: pd.DataFrame, ephemeris_pd:pd.DataFrame, ephemeris_filepath:pd.DataFrame) \
        -> pd.DataFrame:
    """
    Computes satellites and receiver positions

    :param raw_pd:
    :param ephemeris_pd:
    :param ephemeris_filepath:
    :return: SV et RX positions
    """
    # Get SV positions
    pos_pd, pd_gnss_raw = bits.spp.get_position_estimate(pd_gnss_raw=raw_pd,
                                                         pd_ephemeris=ephemeris_pd, ephem_filepath=ephemeris_filepath,
                                                         verbose=True)

    # Clean up
    #pos_pd["unix_time"] = pos_pd["time"].apply(lambda timestamp: timestamp.timestamp_pd.timestamp())
    pos_pd.sort_values("unix_time", inplace=True)
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(lambda timestamp: timestamp.timestamp_pd.timestamp())
    pd_gnss_raw.sort_values("unix_time", inplace=True)

    out_pd = pd.merge_asof(
        pd_gnss_raw,
        pos_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m", "vx_rx_mps", "vy_rx_mps", "vz_rx_mps"]],
        on="unix_time",
        tolerance=1,
        direction="nearest"
    )

    return out_pd


def fast_sv_pos(raw_pd: pd.DataFrame, pos_pd:pd.DataFrame, ephemeris_pd: pd.DataFrame, ephemeris_filepath):
    """
    Computes satellites positions

    :param raw_pd:
    :param pos_pd:
    :param ephemeris_pd:
    :param ephemeris_filepath:
    :return: SV et RX positions
    """
    # Get SV positions
    pd_gnss_raw = bits.spp.get_sv_states(pd_gnss_raw=raw_pd,
                                         pd_ephemeris=ephemeris_pd, ephem_filepath=ephemeris_filepath)

    # Clean up
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(lambda timestamp: timestamp.timestamp_pd.timestamp())
    #pos_pd["unix_time"] = pos_pd["time"].apply(lambda timestamp: timestamp.timestamp_pd.timestamp())
    pd_gnss_raw.sort_values("unix_time", inplace=True)
    pos_pd.sort_values("unix_time", inplace=True)

    # Merge positions
    out_pd = pd.merge_asof(pd_gnss_raw, pos_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                                on="unix_time", direction="nearest").reset_index(drop=True)

    return out_pd

def compute_steering_vector(gnss_data_pd: pd.DataFrame) -> pd.DataFrame:
    """
    :param gnss_data_pd: at least contain receiver and sv positions
    :return:
    """
    # Compute steering vectors
    gnss_data_pd["range"] = np.sqrt(
        (gnss_data_pd["x_sv_m"] - gnss_data_pd["x_rx_m"]) ** 2 +
        (gnss_data_pd["y_sv_m"] - gnss_data_pd["y_rx_m"]) ** 2 +
        (gnss_data_pd["z_sv_m"] - gnss_data_pd["z_rx_m"]) ** 2
    )

    gnss_data_pd["steering_vector_x"] = (gnss_data_pd["x_sv_m"] - gnss_data_pd["x_rx_m"]) / gnss_data_pd["range"]
    gnss_data_pd["steering_vector_y"] = (gnss_data_pd["y_sv_m"] - gnss_data_pd["y_rx_m"]) / gnss_data_pd["range"]
    gnss_data_pd["steering_vector_z"] = (gnss_data_pd["z_sv_m"] - gnss_data_pd["z_rx_m"]) / gnss_data_pd["range"]

    # Compute SV elevation and azimuth
    pvt = gnss_data_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]].drop_duplicates(subset="time", keep="first")
    gnss_data_pd = bits.spp.get_sv_el_az(gnss_data_pd, pvt)


    return gnss_data_pd