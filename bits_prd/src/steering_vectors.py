import bits # Available at https://github.com/enkisaura/Baguette-In-The-Sky.git

import numpy as np
import pandas as pd
from typing import Tuple


def compute_geometry_matrix(raw_pd: pd.DataFrame,
                            ephemeris_pd: pd.DataFrame|None=None, ephemeris_filepath: str|None=None,
                            pos_pd:pd.DataFrame|None=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    if pos_pd is None:
        pos_pd, pd_gnss_raw = slow_sv_pos(raw_pd, ephemeris_pd, ephemeris_filepath) # Also computes approximate position
    else:
        pos_pd, pd_gnss_raw = fast_sv_pos(raw_pd, pos_pd, ephemeris_pd, ephemeris_filepath)

    pd_gnss_raw = pd_gnss_raw.dropna(subset=["e_x"])
    return pos_pd, pd_gnss_raw


def slow_sv_pos(raw_pd: pd.DataFrame, ephemeris_pd:pd.DataFrame|None, ephemeris_filepath:str|None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes satellites and receiver positions without a priori knowledge on receiver position.

    :param raw_pd: BITS raw dataframe
    :param ephemeris_pd: BITS ephemeris dataframe
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :return: BITS raw dataframe with SV and RX positions
    """
    # Get SV positions
    pos_pd, pd_gnss_raw = bits.single_point_positioning.get_position_estimate(pd_gnss_raw=raw_pd,
                                                                              pd_ephemeris=ephemeris_pd,
                                                                              ephem_filepath=ephemeris_filepath,
                                                                              verbose=True)

    # Clean up
    pos_pd.sort_values("time", inplace=True)
    pd_gnss_raw.sort_values("time", inplace=True)

    return pos_pd, pd_gnss_raw


def fast_sv_pos(raw_pd: pd.DataFrame, pos_pd:pd.DataFrame, ephemeris_pd: pd.DataFrame|None,
                ephemeris_filepath:str|None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes satellites positions with a priori knowledge on receiver position.

    :param raw_pd: BITS raw dataframe
    :param pos_pd: BITS PVT dataframe
    :param ephemeris_pd: BITS ephemeris dataframe
    :param ephemeris_filepath: Filepath to Rinex nav ephemeris
    :return: BITS raw dataframe with SV and RX positions
    """
    # Get SV positions
    pd_gnss_raw = bits.sv_model.get_sv_states(pd_gnss_raw=raw_pd, pd_ephemeris=ephemeris_pd,
                                              ephem_filepath=ephemeris_filepath)

    # Clean up
    pd_gnss_raw.sort_values("time", inplace=True)
    pos_pd.sort_values("time", inplace=True)

    # Add steering vectors
    pvt_time_list = pos_pd["time"].tolist()
    raw_pd_list = []
    for raw_time, group in pd_gnss_raw.groupby("time"):
        sv_position = group[["x_sv_m", "y_sv_m", "z_sv_m"]].to_numpy()

        pvt_closest_time = min(pvt_time_list, key=lambda d: abs(d - raw_time))
        pvt_at_timestamp = pos_pd[pos_pd["time"] == pvt_closest_time]
        rx_pos = pvt_at_timestamp[["x_rx_m", "y_rx_m", "z_rx_m"]].to_numpy()

        geometry_matrix_np = bits.single_point_positioning.compute_geometry_matrix(sv_position, rx_pos)

        group["e_x"] = geometry_matrix_np[:, 0]
        group["e_y"] = geometry_matrix_np[:, 1]
        group["e_z"] = geometry_matrix_np[:, 2]
        group["e_b"] = geometry_matrix_np[:, 3]

        raw_pd_list.append(group)

    pd_gnss_raw = pd.concat(raw_pd_list, ignore_index=True)

    return pos_pd, pd_gnss_raw