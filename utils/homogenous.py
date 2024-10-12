"""
Helper functions for homogeneous transformation matrices

SceneFun3D Toolkit
"""

import numpy as np

def rot(H):
    """
    Extracts the 3x3 rotation matrix from a 4x4 homogeneous transformation matrix.

    Args:
        H (numpy.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: The 3x3 rotation matrix from the upper-left submatrix of H.
    """
    return H[:3, :3]

def trans(H):
    """
    Extracts the 3x1 translation vector from a 4x4 homogeneous transformation matrix.

    Args:
        H (numpy.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: The 3x1 translation vector from the rightmost column of H.
    """
    return H[:3, 3]

def inverse(H):
    """
    Computes the inverse of a 4x4 homogeneous transformation matrix.

    The inverse is computed by transposing the 3x3 rotation matrix and negating the transformed translation vector.

    Args:
        H (numpy.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: The 4x4 inverse of the input homogeneous transformation matrix.
    """
    H_inv = np.eye(4)
    H_inv[0:3, 0:3] = rot(H).T
    H_inv[0:3, 3] = -rot(H).T @ trans(H)
    
    return H_inv