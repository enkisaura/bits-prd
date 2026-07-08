import bits # Available at https://github.com/enkisaura/Baguette-In-The-Sky.git

import numpy as np
import pandas as pd

def compute_geometry_matrix(raw_pd: pd.DataFrame,
                            ephemeris_pd: pd.DataFrame|None=None, ephemeris_filepath: str|None=None,
                            pos_pd:pd.DataFrame|None=None) -> pd.DataFrame:
    """
    Computes the geometry matrix from raw data and ephemeris data.

    A priori knowledge on receiver position can be given with argument "pos_pd". Otherwise, receiver position will be
    computed using BITS single point positioning.

    :param raw_pd: BITS raw dataframe
    :param ephemeris_pd: BITS ephemeris dataframe
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :param pos_pd: BITS PVT dataframe
    :return: BITS raw dataframe with geometry matrix
    """

    if "unix_time" not in raw_pd.columns:
        raw_pd["unix_time"] = raw_pd["time"].apply(lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())

    if pos_pd is not None:
        if "unix_time" not in pos_pd.columns:
            pos_pd["unix_time"] = pos_pd["time"].apply(lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())

    if pos_pd is None:
        out_pd = slow_sv_pos(raw_pd, ephemeris_pd, ephemeris_filepath) # Also computes approximate position
    else:
        out_pd = fast_sv_pos(raw_pd, pos_pd, ephemeris_pd, ephemeris_filepath)

    # Compute steering vectors
    out_pd = _compute_steering_vector(out_pd)

    return out_pd


def slow_sv_pos(raw_pd: pd.DataFrame, ephemeris_pd:pd.DataFrame|None, ephemeris_filepath:str|None) \
        -> pd.DataFrame:
    """
    Computes satellites and receiver positions without a priori knowledge on receiver position.

    :param raw_pd: BITS raw dataframe
    :param ephemeris_pd: BITS ephemeris dataframe
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :return: BITS raw dataframe with SV and RX positions
    """
    # Get SV positions
    pos_pd, pd_gnss_raw = bits.spp.get_position_estimate(pd_gnss_raw=raw_pd,
                                                         pd_ephemeris=ephemeris_pd, ephem_filepath=ephemeris_filepath,
                                                         verbose=True)

    # Clean up
    pos_pd.sort_values("unix_time", inplace=True)
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(lambda timestamp: timestamp.timestamp_pd.timestamp())
    pd_gnss_raw.sort_values("unix_time", inplace=True)

    # Add computed receiver position
    out_pd = pd.merge_asof(
        pd_gnss_raw,
        pos_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m", "vx_rx_mps", "vy_rx_mps", "vz_rx_mps"]],
        on="unix_time",
        tolerance=1,
        direction="nearest"
    )

    return out_pd


def fast_sv_pos(raw_pd: pd.DataFrame, pos_pd:pd.DataFrame, ephemeris_pd: pd.DataFrame|None,
                ephemeris_filepath:str|None) -> pd.DataFrame:
    """
    Computes satellites positions with a priori knowledge on receiver position.

    :param raw_pd: BITS raw dataframe
    :param pos_pd: BITS PVT dataframe
    :param ephemeris_pd: BITS ephemeris dataframe
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :return: BITS raw dataframe with SV and RX positions
    """
    # Get SV positions
    pd_gnss_raw = bits.spp.get_sv_states(pd_gnss_raw=raw_pd,
                                         pd_ephemeris=ephemeris_pd, ephem_filepath=ephemeris_filepath)

    # Clean up
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(lambda timestamp: timestamp.timestamp_pd.timestamp())
    pd_gnss_raw.sort_values("unix_time", inplace=True)
    pos_pd.sort_values("unix_time", inplace=True)

    # Add a priori knowledge on receiver position
    out_pd = pd.merge_asof(pd_gnss_raw, pos_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                                on="unix_time", direction="nearest").reset_index(drop=True)

    return out_pd

def _compute_steering_vector(gnss_data_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Computes steering vectors.

    :param gnss_data_pd: BITS raw dataframe with SV et RX positions
    :return: BITS raw dataframe with steering vectors
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