from numpy.typing import NDArray
import numpy as np

def vecadd(
    A: NDArray[np.float32], B: NDArray[np.float32], chunked: bool = False
) -> NDArray[np.float32]: ...
