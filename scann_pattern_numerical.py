import numpy as np
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
import tqdm


def refract_beam(b, n, nl, nr):
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


# noinspection PyUnreachableCode
def rot_surfaces(t, n, omega, theta0):
    # calculate the rotation angle
    theta = omega * t + theta0
    # vector of rotation axis
    r = np.array([0, 0, 1])
    # rotate the normal vector with the Rodrigues rotation formula
    rxn = np.cross(r, n)
    n = n * np.cos(theta) + np.sin(theta) * rxn + r * np.dot(r, n) * (1 - np.cos(theta))

    return n


def main():
    #  choose scanning pattern
    repetitive = True  # True: repetitive, False: non-repetitive
    # repetitive = False
    DJI_Zenmuse_L2 = False  # True: DJI Zenmuse L2, False: Livox Mid-40

    # define all the parameters
    # refractive index of the prisms
    if DJI_Zenmuse_L2 is True:
        n1 = 1.51
        n2 = 1.51
        n3 = 1.51
    else:  # Livox Mid-40
        n1 = 1.5095
        n2 = 1.5095
    nAir = 1.0
    # angle of prisms (in radians); calculated from k (ratio of deviation angles)
    if DJI_Zenmuse_L2 is True:
        alpha1_deg = 27.65251  # degrees
        # alpha1_deg = 5.72076  # degrees
        alpha1 = np.deg2rad(alpha1_deg)
        k1 = 1.072
        # k1 = 4.65408
        k2 = 0.100565
        # k2 = -0.16
        alpha2 = k1 * alpha1 * (n1 - 1) / (n2 - 1)
        alpha3 = k2 * alpha1 * (n1 - 1) / (n3 - 1)
    else:  # Livox Mid-40
        alpha1_deg = 18.0  # degrees
        alpha1 = np.deg2rad(alpha1_deg)
        alpha2_deg = 18.0  # degrees
        alpha2 = np.deg2rad(alpha2_deg)
    # rotation velocity of the prisms (in radians per second) and initial rotation angle (in radians)
    # theta01 = np.pi
    theta01 = 0.0
    # theta02 = np.pi
    theta02 = 0.0
    if DJI_Zenmuse_L2 is True:
        # theta03 = np.pi
        theta03 = 0.0
        if repetitive is True:
            omega1_rpm = 3000.0
            M1 = -1.0
            M2 = -3.0
        else:  # non-repetitive
            omega1_rpm = 631.7
            M1 = -(5.0 - 1.0 / (10 + 2 / 3))
            # M1 = -4.8
            M2 = 1.0
        omega1 = omega1_rpm * 2 * np.pi / 60  # rad per sec
        omega2 = omega1 * M1
        omega3 = omega1 * M2
    else:  # Livox Mid-40
        omega1_rpm = -4664.0
        omega1 = omega1_rpm * 2 * np.pi / 60  # rad per sec
        omega2_rpm = 7294.0
        omega2 = omega2_rpm * 2 * np.pi / 60  # rad per sec

    # definition of the prism surfaces (1= perpendicular to the rotation axis, 2= angled side of prism)
    # configuration of the prisms: 12 - 21 - 21 !!!check dair1 and dair2 for realistic calculation!!!
    config1 = 12
    config2 = 21
    if DJI_Zenmuse_L2 is True:
        config3 = 21
    # prism 1 (12)
    d1 = 4.0  # mm
    R1 = 20.0  # mm
    # points on the prism surfaces of rotation axis
    P11 = np.array([0, 0, 0])
    z12 = d1 + R1 * np.tan(alpha1)
    P12 = np.array([0, 0, z12])
    # normal vectors to the prism surfaces
    if config1 == 12:
        n11 = np.array([0, 0, 1])
        n12 = np.array([0, np.sin(alpha1), np.cos(alpha1)])
        n12_0 = n12
    elif config1 == 21:
        n11 = np.array([0, -np.sin(alpha1), np.cos(alpha1)])
        n11_0 = n11
        n12 = np.array([0, 0, 1])
    else:
        raise ValueError('Invalid configuration for prism 1')

    # prism 2 (21)
    d2 = 4.0  # mm
    R2 = 20.0  # mm
    # air between prism 1 and prism 2
    dair1 = 3.0 + R1 * np.tan(alpha1) + R2 * np.tan(alpha2)  # mm
    # points on the prism surfaces of rotation axis
    P21 = np.array([0, 0, z12 + dair1])
    z22 = z12 + dair1 + d2 + R2 * np.tan(alpha2)
    P22 = np.array([0, 0, z22])
    # normal vectors to the prism surfaces
    if config2 == 12:
        n21 = np.array([0, 0, 1])
        n22 = np.array([0, np.sin(alpha2), np.cos(alpha2)])
        n22_0 = n22
    elif config2 == 21:
        n21 = np.array([0, -np.sin(alpha2), np.cos(alpha2)])
        n21_0 = n21
        n22 = np.array([0, 0, 1])
    else:
        raise ValueError('Invalid configuration for prism 2')
    if DJI_Zenmuse_L2 is True:
        # prism 3 (21)
        d3 = 1.0  # mm
        R3 = 10.0  # mm
        # air between prism 2 and prism 3
        dair2 = 2.0 + R3 * np.tan(alpha3)  # mm
        # points on the prism surfaces of rotation axis
        P31 = np.array([0, 0, z22 + dair2])
        z32 = z22 + dair2 + d3 + R3 * np.tan(alpha3)
        P32 = np.array([0, 0, z32])
        # normal vectors to the prism surfaces
        if config3 == 12:
            n31 = np.array([0, 0, 1])
            n32 = np.array([0, np.sin(alpha3), np.cos(alpha3)])
            n32_0 = n32
        elif config3 == 21:
            n31 = np.array([0, -np.sin(alpha3), np.cos(alpha3)])
            n31_0 = n31
            n32 = np.array([0, 0, 1])
        else:
            raise ValueError('Invalid configuration for prism 3')

    # distance to observation plane
    D = 70000.0  # mm
    # D = 50000.0  # mm
    # observation plane
    if DJI_Zenmuse_L2 is True:
        P4 = np.array([0, 0, z32 + D])
    else:  # Livox Mid-40
        P4 = np.array([0, 0, z22 + D])  # to show the pattern when using prism 1 and prism 2 (also change Pobs)
        # P4 = np.array([0, 0, z12 + D])  # to show the pattern when only using prism 1 (also change Pobs)
    n4 = np.array([0, 0, 1])

    # beam line
    P00 = np.array([0, 0, -1])
    if DJI_Zenmuse_L2 is True:
        lims = 0.0268873  # mm  # half the distance between the most left and right beam at the source
        b00s = [np.array([a, 0, 1])/np.linalg.norm(np.array([a, 0, 1])) for a
                in np.linspace(-lims, lims, 6)
                ]  # beam direction vectors when using 6 beams
        if repetitive is True:
            ts = np.arange(0, 1 * 0.02, 1 / 240000)  # repetitive
        else:  # non-repetitive
            ts = np.arange(0, 1.0/(10+2/3)*1.0456543, 1/240000)  # only one rotation
            # ts = np.arange(0, 1 * 1.0456543, 1 / 240000)
    else:  # Livox Mid-40
        b00 = np.array([0, 0, 1])  # beam direction vector when using only one beam
        ts = np.arange(0, 0.1, 1/100000)

    # list for the intersection point coordinates with the observation plane
    if DJI_Zenmuse_L2 is True:  # 6 beams
        PobsX = np.empty((ts.shape[0]*6,), dtype=float)
        PobsY = np.empty((ts.shape[0]*6,), dtype=float)
        PobsZ = np.empty((ts.shape[0]*6,), dtype=float)
    else:  # Livox Mid-40; 1 beam
        PobsX = np.empty((ts.shape[0],), dtype=float)
        PobsY = np.empty((ts.shape[0],), dtype=float)
        PobsZ = np.empty((ts.shape[0],), dtype=float)

    # loop over the integration time
    for tix, t in enumerate(tqdm.tqdm(ts)):
        # rotate the normal vectors of the prism surfaces (only the angular surfaces(Nr.2); the others stay the same)
        if config1 == 12:
            n12 = rot_surfaces(t, n12_0, omega1, theta01)
        else:
            n11 = rot_surfaces(t, n11_0, omega1, theta01)
        if config2 == 12:
            n22 = rot_surfaces(t, n22_0, omega2, theta02)
        else:
            n21 = rot_surfaces(t, n21_0, omega2, theta02)
        if DJI_Zenmuse_L2 is True:
            if config3 == 12:
                n32 = rot_surfaces(t, n32_0, omega3, theta03)
            else:
                n31 = rot_surfaces(t, n31_0, omega3, theta03)
        # refract the beam at the prism surfaces and the observation plane
        if DJI_Zenmuse_L2 is True:
            for bix, b00 in enumerate(b00s):
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
                # append the intersection point with the observation plane (6 beams)
                PobsX[tix*6+bix] = Pobs[0]
                PobsY[tix*6+bix] = Pobs[1]
                PobsZ[tix*6+bix] = Pobs[2]
        else:  # Livox Mid-40
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

            # Refraction at observation plane
            Pobs = intersect((Pb22, b22), (P4, n4))  # to show the pattern when using prism 1 and prism 2 (also change P4)
            # Pobs = intersect((Pb12, b12), (P4, n4))  # to show the pattern when only using prism 1 (also change P4)
            # append the intersection point with the observation plane (1 beam)
            PobsX[tix] = Pobs[0]
            PobsY[tix] = Pobs[1]
            PobsZ[tix] = Pobs[2]

    # show intersection with observation plane as plot
    plt.figure(figsize=(10, 10), dpi=300)
    marker = matplotlib.markers.MarkerStyle(marker='o')
    marker.fillstyle = 'full'
    # calculate the FOV angles phi and kappa
    phi = np.atan2(PobsX, D) * 180/np.pi
    kappa = np.atan2(PobsY, np.sqrt(np.square(PobsX) + D**2)) * 180/np.pi
    plt.axis('equal')
    if DJI_Zenmuse_L2 is True:
        plt.scatter(PobsX, PobsY, marker=marker, s=0.01, c=ts.repeat(6), cmap='viridis')
        # plt.scatter(phi, kappa, marker=marker, s=0.01, c=ts.repeat(6), cmap='viridis')
        plt.title(f'M1={M1} M2={M2} k1={k1} k2={k2} alpha1={alpha1_deg} omega1={omega1_rpm} lims={lims} '
                  f'd1={d1}\nd2={d2} d3={d3} R1={R1} R2={R2} R3={R3} dair1={dair1:.4f} dair2={dair2:.4f} D={D} t={ts[-1]:.7f}')
    else:  # Livox Mid-40
        # plt.scatter(PobsX, PobsY, marker=marker, s=0.01, c=ts.repeat(1), cmap='viridis')
        plt.scatter(phi, kappa, marker=marker, s=0.1, color='blue')
        plt.title(f'alpha1={alpha1_deg} alpha2={alpha2_deg} omega1={omega1_rpm} omega2={omega2_rpm} n1={n1} n2={n2}\n'
                  f'd1={d1} d2={d2} R1={R1} R2={R2} dair1={dair1:.5f} D={D} t={ts[-1]}')
    # plt.xlim(-4000, 4000)
    # plt.ylim(-3000, 3000)
    # plt.xlim(-7500, 7500)
    # plt.ylim(-70000, -45000)

    # print the maximal phi and kappa values
    """print(f'Maximal phi: {np.max(phi)}')
    print(f'Maximal kappa: {np.max(kappa)}')"""
    # get the phi values for kappa = 0 till 0.01
    kappa_0 = 0
    phi_0 = phi[np.where((kappa >= kappa_0) & (kappa <= kappa_0 + 0.1))]
    # print(f'Phi values for kappa = 0: {phi_0}')
    # print the maximal and minimal phi value for kappa = 0
    """print(f'Maximal phi for kappa = 0: {np.max(phi_0)}')
    print(f'Minimal phi for kappa = 0: {np.min(phi_0)}')"""
    plt.text(0.01, 0.01, f'Max $\phi$: {np.max(phi):.5f}\n'
                       f'Max $\kappa$: {np.max(kappa):.5f}\n'
                       f'$\phi$_0[0]: {phi_0[0]:.5f}\n'
                       f'$\phi$_0[5]: {phi_0[5]:.5f}',
             horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.show()

if __name__ == '__main__':
    main()
