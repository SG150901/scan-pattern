import numpy as np
import matplotlib.pyplot as plt
import tqdm

def refract_beam(v, n, nl, nr):
    # get the direction vector of the beam
    b = v
    # get the normal vector of the plane
    n = n
    # get the refractive index of the left medium
    nl = nl
    # get the refractive index of the right medium
    nr = nr

    # calculate the angle of incidence
    cos_delta_i = np.dot(b, -n)  # b and n are normalized
    # calculate the angle of refraction
    cos_delta_r = np.sqrt(1 - (nl / nr)**2 * (1 - cos_delta_i**2))
    # calculate the direction vector of the refracted beam
    b = nl / nr * b + (nl / nr * cos_delta_i - cos_delta_r) * n

    return b

def intersect(g, e):
    # get the point and the direction vector on the line g
    Pg, vg = g
    # get the point and the normal vector on the plane e
    Pe, ne = e

    # calculate the intersection point
    n = np.dot(vg, ne)
    if n == 0:
        return None
    P = Pg + np.dot(Pe - Pg, ne) / n * vg

    return P

def rot_surfaces(t, n, omega, theta0):
    # calculate the rotation angle
    theta = omega * t + theta0
    # vector of rotation axis
    r = np.array([0, 0, 1])
    # rotate the normal vector with the Rodrigues rotation formula
    rxn = np.cross(r, n)
    n = n * np.cos(theta) + np.sin(theta) * rxn + r * np.dot(r, n) * (1 - np.cos(theta))
    # print(n)

    return n

def main():
    # define all the parameters
    # integration time [s]
    t = 0.00
    T = 0.01
    # refractive index of the prisms
    n1 = 1.5
    n2 = 1.5
    n3 = 1.5
    nAir = 1.0
    # angle of prisms (in radians); calculated from k (ratio of deviation angles)
    alpha1_deg = 18.8
    alpha1 = np.deg2rad(alpha1_deg)
    k1 = 1.0
    k2 = 0.7
    alpha2 = k1 * alpha1 * (n1 - 1) / (n2 - 1)
    alpha3 = k2 * alpha1 * (n1 - 1) / (n3 - 1)
    # rotation velocity of the prisms (in radians per  second) and initial rotation angle (in radians)
    theta01 = 0.0
    theta02 = 0.0
    theta03 = 0.0
    omega1_rpm = 4000
    omega1 = omega1_rpm * 2 * np.pi / 60  # rad per sec
    M1 = -2.0
    M2 = 0.15
    omega2 = omega1 * M1
    omega3 = omega1 * M2

    # definition of the prism surfaces
    # prism 1 (12)
    d1 = 2.0  # mm
    R1 = 20.0  # mm
    # points on the prism surfaces of rotation axis
    P11 = np.array([0, 0, 0])
    z12 = d1 + R1 * np.tan(alpha1)
    P12 = np.array([0, 0, d1 + z12])
    # normal vectors to the prism surfaces
    n11 = np.array([0, 0, 1])
    n12 = np.array([0, np.sin(alpha1), np.cos(alpha1)])
    # air between prism 1 and prism 2
    dair1 = 10  # mm

    # prism 2 (21)
    d2 = 2.0  # mm
    R2 = 20.0  # mm
    # points on the prism surfaces of rotation axis
    P21 = np.array([0, 0, z12 + dair1])
    z22 = z12  + dair1 + d2 + R2 * np.tan(alpha2)
    P22 = np.array([0, 0, z22])
    # normal vectors to the prism surfaces
    n21 = np.array([0, -np.sin(alpha2), np.cos(alpha2)])
    n22 = np.array([0, 0, 1])
    # air between prism 2 and prism 3
    dair2 = 600  # mm

    # prism 3 (21)
    d3 = 2.0  # mm
    R3 = 20.0  # mm
    # points on the prism surfaces of rotation axis
    P31 = np.array([0, 0, z22 + dair2])
    z32 = z22 + dair2 + d3 + R3 * np.tan(alpha3)
    P32 = np.array([0, 0, z32])
    # normal vectors to the prism surfaces
    n31 = np.array([0, -np.sin(alpha3), np.cos(alpha3)])
    n32 = np.array([0, 0, 1])

    # distance to observation plane
    D = 1000  # mm
    # observation plane
    P4 = np.array([0, 0, z32 + D])
    n4 = np.array([0, 0, 1])

    # beam line
    P00 = np.array([0, 0, -1])
    b00 = np.array([0, 0, 1])

    # list for the intersection point coordinates with the observation plane
    PobsX = []
    PobsY = []
    PobsZ = []

    # loop over the integration time
    for t in tqdm.tqdm(np.arange(0, 0.0003333, 1/200000000)):
        # Refraction at surface 11
        Pb11 = intersect((P00, b00), (P11, n11))
        b11 = refract_beam(b00, n11, nAir, n1)

        # Refraction at surface 12
        Pb12 = intersect((Pb11, b11), (P12, n12))
        b12 = refract_beam(b11, n12, n1, nAir)

        # Refraction at surface 21
        Pb21 = intersect((Pb12, b12), (P21, n21))
        b21 = refract_beam(b12, n21, nAir, n2)

        # Refraction at surface 22
        Pb22 = intersect((Pb21, b21), (P22, n22))
        b22 = refract_beam(b21, n22, n2, nAir)

        # Refraction at surface 31
        Pb31 = intersect((Pb22, b22), (P31, n31))
        b31 = refract_beam(b22, n31, nAir, n3)

        # Refraction at surface 32
        Pb32 = intersect((Pb31, b31), (P32, n32))
        b32 = refract_beam(b31, n32, n3, nAir)

        # Refraction at observation plane
        Pobs = intersect((Pb32, b32), (P4, n4))
        # Pobs = intersect((Pb12, b12), (P4, n4))
        # Pobs = intersect((Pb22, b22), (P4, n4))
        # append the intersection point with the observation plane
        PobsX.append(Pobs[0])
        PobsY.append(Pobs[1])
        PobsZ.append(Pobs[2])

        """# plot the beam line
        plt.plot([P00[2], Pb11[2], Pb12[2], Pb21[2], Pb22[2], Pb31[2], Pb32[2], Pobs[2]], [P00[1], Pb11[1], Pb12[1], Pb21[1], Pb22[1], Pb31[1], Pb32[1], Pobs[1]], 'o')
        plt.axis('equal')
        plt.show()"""

        # increase the time

        # rotate the normal vectors of the prism surfaces (only the angular surfaces(Nr.2); the others stay the same)
        n12 = rot_surfaces(t, n12, omega1, theta01)
        n21 = rot_surfaces(t, n21, omega2, theta02)
        n31 = rot_surfaces(t, n31, omega3, theta03)

    # show intersection with observation plane as plot
    plt.plot(PobsX, PobsY, 'o', markersize=0.1)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
    """g = (np.array([1, 2, 3]), np.array([1, 1, 0]))
    e = (np.array([1, 3, 3]), np.array([0.4, -0.8, 0]))
    P = intersect(g, e)
    print(P)
    plt.plot(g[0][0], g[0][1], 'o')
    plt.plot(e[0][0], e[0][1], 'o')
    plt.plot([g[0][0], g[0][0]+g[1][0]], [g[0][1], g[0][1] + g[1][1]], 'b--')
    plt.plot([e[0][0], e[0][0]+e[1][0]], [e[0][1], e[0][1] + e[1][1]], 'r--')
    plt.plot(P[0], P[1], 'o')
    plt.axis('equal')
    plt.show()"""


