import numpy as np

def wls(Y:np.ndarray, G:np.ndarray, W:np.ndarray) -> None | tuple[np.ndarray, np.ndarray]:
    GtW = G.T @ W  # (m, n)
    GtWG = GtW @ G  # (m, m)
    GtWy = GtW @ Y  # (m,)

    try:
        cov_b = np.linalg.inv(GtWG)  # (m, m)
    except np.linalg.LinAlgError:
        return None

    b_hat = cov_b @ GtWy

    return b_hat, cov_b