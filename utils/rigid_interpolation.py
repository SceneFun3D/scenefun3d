"""
Implements rigid body motion interpolation in 3D

Usage: Use the function rigid_interp_geodesic() or rigid_interp_split() as shown in the demo code

SceneFun3D Toolkit
"""

import numpy as np

def InverseRigid(H):
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    invH = np.eye(4)
    invH[0:3, 0:3] = H_R.T
    invH[0:3, 3] = -H_R.T @ H_T

    return invH

def Exp(t, theta, S):
    angle = t * theta
    thetaSqr = theta * theta

    if theta > 0:
        return np.eye(3) + (np.sin(angle) / theta) * S + ((1 - np.cos(angle)) / thetaSqr) * S @ S
    else:
        return np.eye(3)

def Log(R):
    S = np.array((3, 3))

    arg = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1) # in [-1,1]

    if arg > -1:
        if arg < 1:
            # 0 < angle < pi
            angle = np.arccos(arg)
            sinAngle = np.sin(angle)

            c = 0.5 * angle / sinAngle
            S = c * (R - R.T)
        else: # arg = 1, angle = 0
            # R is the identity matrix and S is the zero matrix
            S = np.zeros((3, 3))
    else: # arg = -1, angle = pi
        # Knowing R+I is symmetric and wanting to avoid bias, we use
        # ( R(i,j) + R(j,i) ) / 2 for the off−diagonal entries rather than R(i,j)
        s = np.zeros((3, 1))
        if R[0, 0] >= R[1, 1]:
            if R[0, 0] >= R[2, 2]:
                # r00 is the maximum diagonal term
                s[0] = R[0, 0] + 1
                s[1] = 0.5 * (R[0, 1] + R[1, 0])
                s[2] = 0.5 * (R[0, 2] + R[2, 0])
            else:
                # r22 is the maximum diagonal term
                s[0] = 0.5 * (R[2, 0] + R[0, 2])
                s[1] = 0.5 * (R[2, 1] + R[1, 2])
                s[2] = R[2, 2] + 1
        else:
            if R[1, 1] >= R[2, 2]:
                # r11 is the maximum diagonal term
                s[0] = 0.5 * (R[1, 0] + R[0, 1])
                s[1] = R[1, 1] + 1
                s[2] = 0.5 * (R[1, 2] + R[2, 1])

            else:
                # r22 is the maximum diagonal term
                s[0] = 0.5 * (R[2, 0] + R[0, 2])
                s[1] = 0.5 * (R[2, 1] + R[1, 2])
                s[2] = R[2, 2] + 1

        length = np.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])

        if length > 0:
            adjust = np.pi * np.sqrt(0.5) / length
            s = adjust * s

        else:
            s = np.zeros((3, 1))


        S[0, 0] = 0.0
        S[0, 1] = -s[2]
        s[0, 2] = s[1]
        S[1, 0] = s[2]
        S[1, 1] = 0.0
        S[1, 2] = -s[0]
        S[2, 0] = -s[1]
        S[2, 1] = s[0]
        S[2, 2] = 0.0

    assert S.shape == (3, 3)
    return S

def computeTTimesV(t, theta, S):
    if theta > 0:
        angle = t * theta
        thetaSqr = theta * theta 
        thetaCub = theta * thetaSqr

        c0 = (1 - np.cos(angle)) / thetaSqr
        c1 = (angle - np.sin(angle)) / thetaCub

        return t * np.eye(3) + c0 * S + c1 * S @ S
    else:
        return t * np.eye(3)
    
def computeInverseV1(theta, S):
    if theta > 0:
        thetaSqr = theta * theta
        c = (1 - (theta * np.sin(theta)) / (2 * (1 - np.cos(theta)))) / thetaSqr

        return np.eye(3) - 0.5 * S + c * S @ S
    else:
        return np.eye(3)
    
def GeodesicPath(t, H0, H1):

    # If you plan on calling Geodesic Path for the same H0 and H1 but for multiple
    # t−values, the following terms can be precomputed and cached for use by the
    # last block of code
    H = H1 @ InverseRigid(H0)
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    S = Log(H_R)

    s0 = S[2, 1]
    s1 = S[0, 2]
    s2 = S[1, 0]
    theta = np.sqrt(s0*s0 + s1*s1 + s2*s2)
    invV1 = computeInverseV1(theta, S)
    U = invV1 @ H_T
    #### until here the terms can be precomputed for multiple t-values

    interpR = Exp(t, theta, S)
    interpTTimesV = computeTTimesV(t, theta, S)

    interpH = np.eye(4)
    H0_R = H0[0:3, 0:3]
    H0_T = H0[0:3, 3]
    interpH[0:3, 0:3] = interpR @ H0_R
    interpH[0:3, 3] = interpR @ H0_T + interpTTimesV @ U

    return interpH

"""
    Description: Performs interpolation to find the homogenous matrix (4x4) in the desired timestep

    Input:
        * t: desired timestep
        * H0: homogenous matrix (4x4) describing the motion in timestep t0
        * t0: timestep corresponding to H0
        * H1: homogenous matrix (4x4) describing the motion in timestep t1
        * t0: timestep corresponding to H1
    
    Output:
        * H: interpolated homogenous matrix (4x4) describing the motion in timestep t
"""
def rigid_interp_geodesic(t, H0, t0, H1, t1):
    """
    Performs rigid body motion interpolation in SE(3). See https://www.geometrictools.com/Documentation/InterpolationRigidMotions.pdf.

    Args:
        t (float): desired timestep
        H0 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t0
        t0 (float): timestep corresponding to H0
        H1 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t1
        t1 (float): timestep corresponding to H1

    Returns:
        (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t  
    
    """

    # map t in the interval [0, 1]
    slope = (1.0 - 0.0) / (t1 - t0)
    t_ = 0.0 + slope * (t - t0)

    return GeodesicPath(t_, H0, H1)

def rigid_interp_split(t, H0, t0, H1, t1):
    """
    Performs rigid body motion interpolation in SO(3) x R^3. See https://www.adrian-haarbach.de/interpolation-methods/doc/haarbach2018survey.pdf.

    Args:
        t (float): desired timestep
        H0 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t0
        t0 (float): timestep corresponding to H0
        H1 (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t1
        t1 (float): timestep corresponding to H1

    Returns:
        (numpy.ndarray): homogenous matrix (4x4) describing the motion in timestep t  
    
    """

    # map t in the interval [0, 1]
    slope = (1.0 - 0.0) / (t1 - t0)
    t_ = 0.0 + slope * (t - t0)

    H0_R = H0[0:3, 0:3]
    H0_T = H0[0:3, 3]
    H0_new = np.eye(4)
    H0_new[0:3, 0:3] = H0_R

    H1_R = H1[0:3, 0:3]
    H1_T = H1[0:3, 3]
    H1_new = np.eye(4)
    H1_new[0:3, 0:3] = H1_R

    interpH = np.eye(4)

    interpH[0:3, 0:3] = GeodesicPath(t_, H0_new, H1_new)[0:3, 0:3]

    interpH[0:3, 3] = H0_T + t_ * (H1_T - H0_T)

    return interpH


if __name__ == "__main__":

    # example code
    H0 = np.eye(4)
    t0 = 100

    H1 = np.eye(4)
    # add a translation to H0
    H1[0,3] = 5
    H1[1,3] = 2
    t1 = 200

    t = 150 # desired timestep

    print(H0)
    print(H1)
    print(rigid_interp_split(t, H0, t0, H1, t1))
