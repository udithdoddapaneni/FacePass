import numpy as np

def similarity(v1:np.ndarray, v2:np.ndarray):
    num = (v1 @ v2.T).item()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num/denom