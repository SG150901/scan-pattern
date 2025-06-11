import numpy as np

class RisleyBeamDeflector2():

    cached_emitterRelativeAttitude = np.array([0.0, 0.0, 0.0])

    # parmeters for the different scanner types and patterns

    """# L2 repetitive pattern
    refraction_index_prism1 = 1.51
    refraction_index_prism2 = 1.51
    refraction_index_prism3 = 1.51
    refraction_index_air = 1.0
    prism1_speed = 3000.0  # rot/min
    prism2_speed = -3000.0  # rot/min
    prism3_speed = -9000.0  # rot/min
    prism1_angle = 27.65251  # deg
    prism2_angle = prism1_angle * 1.072  # deg
    prism3_angle = prism1_angle * 0.100565  # deg
    prism1_thickness_base = 1.0  # mm
    prism2_thickness_base = 1.0  # mm
    prism3_thickness_base = 1.0  # mm
    prism1_radius = 10.0  # mm
    prism2_radius = 10.0  # mm
    prism3_radius = 10.0  # mm
    distance_between_prisms1and2 = 2.0  # mm
    distance_between_prisms2and3 = 2.0  # mm
    distance_to_observation_plane = 70000.0  # mm
    lims = 0.0268873  # mm
    number_of_beams = 6
    cachedDeltaT = 1.0 / 240000.0  # sec (240kHz)"""

    """# L2 non-repetitive pattern
    refraction_index_prism1 = 1.51
    refraction_index_prism2 = 1.51
    refraction_index_prism3 = 1.51
    refraction_index_air = 1.0
    prism1_speed = 631.7  # rot/min
    prism2_speed = prism1_speed * -(5.0 - 1.0 / (10 + 2 / 3))  # rot/min
    prism3_speed = prism1_speed  # rot/min
    prism1_angle = 27.65251  # deg
    prism2_angle = prism1_angle * 1.072  # deg
    prism3_angle = prism1_angle * 0.100565  # deg
    prism1_thickness_base = 1.0  # mm
    prism2_thickness_base = 1.0  # mm
    prism3_thickness_base = 1.0  # mm
    prism1_radius = 10.0  # mm
    prism2_radius = 10.0  # mm
    prism3_radius = 10.0  # mm
    distance_between_prisms1and2 = 2.0  # mm
    distance_between_prisms2and3 = 2.0  # mm
    distance_to_observation_plane = 70000.0  # mm
    lims = 0.0268873  # mm
    number_of_beams = 6
    cachedDeltaT = 1.0 / 240000.0  # sec (240kHz)"""

    """# Livox Mid-40
    refraction_index_prism1 = 1.5095
    refraction_index_prism2 = 1.5095
    refraction_index_prism3 = 1.0
    refraction_index_air = 1.0
    prism1_speed = -4664.0  # rot/min
    prism2_speed = 7294.0  # rot/min
    prism3_speed = 0.0  # rot/min
    prism1_angle = 18.0  # deg
    prism2_angle = 18.0  # deg
    prism3_angle = 0.0  # deg
    prism1_thickness_base = 4.0  # mm
    prism2_thickness_base = 4.0  # mm
    prism3_thickness_base = 0.0  # mm
    prism1_radius = 20.0  # mm
    prism2_radius = 20.0  # mm
    prism3_radius = 20.0  # mm
    distance_between_prisms1and2 = 3.0  # mm
    distance_between_prisms2and3 = 0.0  # mm
    distance_to_observation_plane = 70000.0  # mm
    lims = 0.0  # mm
    number_of_beams = 1
    cachedDeltaT = 1.0 / 100000.0  # sec (100000 pts/sec)"""

    def __init__(self,
                 refraction_index_prism1: float,
                 refraction_index_prism2: float,
                 refraction_index_prism3: float,
                 refraction_index_air: float,
                 prism1_speed: float,
                 prism2_speed: float,
                 prism3_speed: float,
                 prism1_angle: float,
                 prism2_angle: float,
                 prism3_angle: float,
                 prism1_thickness_base: float,
                 prism2_thickness_base: float,
                 prism3_thickness_base: float,
                 prism1_radius: float,
                 prism2_radius: float,
                 prism3_radius: float,
                 distance_between_prisms1and2: float,
                 distance_between_prisms2and3: float,
                 distance_to_observation_plane: float,
                 lims: float,
                 number_of_beams: int,
                 cachedDeltaT: float) -> None:  # Konstruktor (implementiert AbstractBeamDeflector(double scanAngleMax_rad, double scanFreqMax_Hz, double scanFreqMin_Hz))
        self.refraction_index_prism1 = refraction_index_prism1
        self.refraction_index_prism2 = refraction_index_prism2
        self.refraction_index_prism3 = refraction_index_prism3
        self.refraction_index_air = refraction_index_air
        self.prism1_speed = prism1_speed
        self.prism2_speed = prism2_speed
        self.prism3_speed = prism3_speed
        self.prism1_angle = prism1_angle
        self.prism2_angle = prism2_angle
        self.prism3_angle = prism3_angle
        self.prism1_thickness_base = prism1_thickness_base
        self.prism2_thickness_base = prism2_thickness_base
        self.prism3_thickness_base = prism3_thickness_base
        self.prism1_radius = prism1_radius
        self.prism2_radius = prism2_radius
        self.prism3_radius = prism3_radius
        self.distance_between_prisms1and2 = distance_between_prisms1and2
        self.distance_between_prisms2and3 = distance_between_prisms2and3
        self.distance_to_observation_plane = distance_to_observation_plane
        self.lims = lims
        self.number_of_beams = number_of_beams
        self.cachedDeltaT = cachedDeltaT

        # sachen berechnen die sich (dann) nicht mehr ändern (Hilfsgrößen, ...)
        self.cachedPrism1NormalVector1 = np.array([0.0, 0.0, 1.0])
        self.cachedPrism1NormalVector2 = np.array([0.0, np.sin(np.radians(self.prism1_angle)), np.cos(np.radians(self.prism1_angle))])
        self.cachedPrism1NormalVector2Original = self.cachedPrism1NormalVector2.copy()
        self.cachedPrism2NormalVector1 = np.array([0.0, -np.sin(np.radians(self.prism2_angle)), np.cos(np.radians(self.prism2_angle))])
        self.cachedPrism2NormalVector1Original = self.cachedPrism2NormalVector1.copy()
        self.cachedPrism2NormalVector2 = np.array([0.0, 0.0, 1.0])
        self.cachedPrism3NormalVector1 = np.array([0.0, -np.sin(np.radians(self.prism3_angle)), np.cos(np.radians(self.prism3_angle))])
        self.cachedPrism3NormalVector1Original = self.cachedPrism3NormalVector1.copy()
        self.cachedPrism3NormalVector2 = np.array([0.0, 0.0, 1.0])
        self.cachedObservationPlaneNormalVector = np.array([0.0, 0.0, 1.0])
        self.cachedBeamDirectionVectors = np.array([[a, 0.0, 1.0]/np.linalg.norm(np.array([a, 0.0, 1.0])) for a
                                           in np.linspace(-self.lims, self.lims, self.number_of_beams)])

        self.cachedPrism1ThicknessSlopedZAxis = prism1_radius * np.tan(np.radians(prism1_angle))
        self.cachedPrism2ThicknessSlopedZAxis = prism2_radius * np.tan(np.radians(prism2_angle))
        self.cachedPrism3ThicknessSlopedZAxis = prism3_radius * np.tan(np.radians(prism3_angle))

        self.cachedBeamZAxisPoint = np.array([0.0, 0.0, -1.0])
        self.cachedPrism1ZAxisPoint1 = np.array([0.0, 0.0, 0.0])
        self.ZAxisCoordinate = self.prism1_thickness_base + self.cachedPrism1ThicknessSlopedZAxis
        self.cachedPrism1ZAxisPoint2 = np.array([0.0, 0.0, self.ZAxisCoordinate])
        self.ZAxisCoordinate += (self.distance_between_prisms1and2 + self.cachedPrism1ThicknessSlopedZAxis
                                 + self.cachedPrism2ThicknessSlopedZAxis)
        self.cachedPrism2ZAxisPoint1 = np.array([0.0, 0.0, self.ZAxisCoordinate])
        self.ZAxisCoordinate += self.prism2_thickness_base + self.cachedPrism2ThicknessSlopedZAxis
        self.cachedPrism2ZAxisPoint2 = np.array([0.0, 0.0, self.ZAxisCoordinate])
        self.ZAxisCoordinate += (self.distance_between_prisms2and3 + self.cachedPrism3ThicknessSlopedZAxis)
        self.cachedPrism3ZAxisPoint1 = np.array([0.0, 0.0, self.ZAxisCoordinate])
        self.ZAxisCoordinate += self.prism3_thickness_base + self.cachedPrism3ThicknessSlopedZAxis
        self.cachedPrism3ZAxisPoint2 = np.array([0.0, 0.0, self.ZAxisCoordinate])
        self.ZAxisCoordinate += self.distance_to_observation_plane
        self.cachedObservationPlaneZAxisPoint = np.array([0.0, 0.0, self.ZAxisCoordinate])

    """def clone(self, other) -> RisleyBeamDeflector2:  # not implemented in python
        pass"""

    def applySettings(self, settings):
        self.active = settings.active
        # ... (siehe ScannerSettings.h)


    def lastPulseLeftDevice(self):  # bei diesem Scannertyp immer True
        return True

    def restartDeflector(self):
        # t auf 0 setzen
        self.t = 0.0  # sec

    def doSimStep(self):
        # hier vektor berechnen und abspeichern

        # calculate the rotation of a prism normal vector based on the prism speed and time
        def rot_surfaces(t, n, omega):
            # calculate the rotation angle
            theta = omega * t
            # vector of rotation axis
            r = np.array([0, 0, 1])
            # rotate the normal vector with the Rodrigues rotation formula
            rxn = np.cross(r, n)
            n = n * np.cos(theta) + np.sin(theta) * rxn + r * np.dot(r, n) * (1 - np.cos(theta))

            return n

        # intersect the beam with the prism surface or the observation plane
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

        # refract the beam at the prism surface
        def refract_beam(b, n, nl, nr):
            # calculate the angle of incidence
            cos_delta_i = np.dot(b, -n)  # b and n are normalized
            # calculate the angle of refraction
            cos_delta_r = np.sqrt(1 - (nl / nr) ** 2 * (1 - cos_delta_i ** 2))
            # calculate the direction vector of the refracted beam
            b = nl / nr * b + (nl / nr * cos_delta_i - cos_delta_r) * n

            return b

        # calculate the beam direction vector
        self.t += self.cachedDeltaT

        # rotate the prism normal vectors based on the prism speeds and time
        Prism1NormalVector2 = rot_surfaces(self.t, self.cachedPrism1NormalVector2Original, self.prism1_speed * 2 * np.pi / 60)
        Prism2NormalVector1 = rot_surfaces(self.t, self.cachedPrism2NormalVector1Original, self.prism2_speed * 2 * np.pi / 60)
        Prism3NormalVector1 = rot_surfaces(self.t, self.cachedPrism3NormalVector1Original, self.prism3_speed * 2 * np.pi / 60)

        Pbeamsobs = np.empty(self.number_of_beams, dtype=object)

        for bix, beam00 in enumerate(self.cachedBeamDirectionVectors):
            # calculate the intersection with the first prism
            Pb11 = intersect((self.cachedBeamZAxisPoint, beam00),
                             (self.cachedPrism1ZAxisPoint1, self.cachedPrism1NormalVector1))
            beam11 = refract_beam(beam00, self.cachedPrism1NormalVector1, self.refraction_index_air, self.refraction_index_prism1)
            Pb12 = intersect((Pb11, beam11), (self.cachedPrism1ZAxisPoint2, Prism1NormalVector2))
            beam12 = refract_beam(beam11, Prism1NormalVector2, self.refraction_index_prism1, self.refraction_index_air)
            # calculate the intersection with the second prism
            Pb21 = intersect((Pb12, beam12), (self.cachedPrism2ZAxisPoint1, Prism2NormalVector1))
            beam21 = refract_beam(beam12, Prism2NormalVector1, self.refraction_index_air, self.refraction_index_prism2)
            Pb22 = intersect((Pb21, beam21), (self.cachedPrism2ZAxisPoint2, self.cachedPrism2NormalVector2))
            beam22 = refract_beam(beam21, self.cachedPrism2NormalVector2, self.refraction_index_prism2, self.refraction_index_air)
            # calculate the intersection with the third prism
            Pb31 = intersect((Pb22, beam22), (self.cachedPrism3ZAxisPoint1, Prism3NormalVector1))
            beam31 = refract_beam(beam22, Prism3NormalVector1, self.refraction_index_air, self.refraction_index_prism3)
            Pb32 = intersect((Pb31, beam31), (self.cachedPrism3ZAxisPoint2, self.cachedPrism3NormalVector2))
            beam32 = refract_beam(beam31, self.cachedPrism3NormalVector2, self.refraction_index_prism3, self.refraction_index_air)
            # calculate the intersection with the observation plane
            Pbobs = intersect((Pb32, beam32), (self.cachedObservationPlaneZAxisPoint, self.cachedObservationPlaneNormalVector))
            # store the intersection point
            Pbeamsobs[bix] = Pbobs

        self.cached_emitterRelativeAttitude = Pbeamsobs

    def getEmitterRelativeAttitude(self):
        return self.cached_emitterRelativeAttitude
        # hier rotation ausgeben (ausgehender Beam als Vektor)

    def hasMechanicalError(self):
        return False  # not implemented at this point


if __name__ == '__main__':
    # Livox Mid-40
    refraction_index_prism1 = 1.5095
    refraction_index_prism2 = 1.5095
    refraction_index_prism3 = 1.0
    refraction_index_air = 1.0
    prism1_speed = -4664.0  # rot/min
    prism2_speed = 7294.0  # rot/min
    prism3_speed = 0.0  # rot/min
    prism1_angle = 18.0  # deg
    prism2_angle = 18.0  # deg
    prism3_angle = 0.0  # deg
    prism1_thickness_base = 4.0  # mm
    prism2_thickness_base = 4.0  # mm
    prism3_thickness_base = 0.0  # mm
    prism1_radius = 20.0  # mm
    prism2_radius = 20.0  # mm
    prism3_radius = 20.0  # mm
    distance_between_prisms1and2 = 3.0  # mm
    distance_between_prisms2and3 = 0.0  # mm
    distance_to_observation_plane = 70000.0  # mm
    lims = 0.0  # mm
    number_of_beams = 1
    cachedDeltaT = 1.0 / 100000.0  # sec (100000 pts/sec)

    defl = RisleyBeamDeflector2(refraction_index_prism1, refraction_index_prism2, refraction_index_prism3, refraction_index_air,
                                prism1_speed, prism2_speed, prism3_speed,
                                prism1_angle, prism2_angle, prism3_angle,
                                prism1_thickness_base, prism2_thickness_base, prism3_thickness_base,
                                prism1_radius, prism2_radius, prism3_radius,
                                distance_between_prisms1and2, distance_between_prisms2and3,
                                distance_to_observation_plane, lims, number_of_beams, cachedDeltaT)
    beam_dirs = []
    defl.restartDeflector()
    # steps = 0.02 * 240000  # 0.02 sec at 240000 Hz (L2 repetitive pattern)
    # steps = 1.0456543 * 240000  # 1.0456 sec at 240000 Hz (L2 non-repetitive pattern)
    steps = 0.1 * 100000  # 0.1 sec at 100000 Hz (Livox Mid-40)
    steps = int(round(steps))
    for i in range(steps):
        defl.doSimStep()
        beam_dir = defl.getEmitterRelativeAttitude()
        for bdi in beam_dir:
            beam_dirs.append(bdi)
    beam_dirs = np.array(beam_dirs)

    phi = np.atan2(beam_dirs[:,0], distance_to_observation_plane) * 180 / np.pi
    kappa = np.atan2(beam_dirs[:,1], np.sqrt(np.square(beam_dirs[:,0]) + distance_to_observation_plane ** 2)) * 180 / np.pi

    import matplotlib.pyplot as plt
    import matplotlib
    plt.figure(figsize=(10, 10), dpi=300)
    plt.axis('equal')
    marker = matplotlib.markers.MarkerStyle(marker='o')
    marker.fillstyle = 'full'
    plt.scatter(beam_dirs[:,0], beam_dirs[:,1], marker=marker, s=0.01, c=list(range(len(beam_dirs))), cmap='viridis')
    plt.text(0.01, 0.01, f'Max $\phi$: {np.max(phi):.5f}\n'
                         f'Max $\kappa$: {np.max(kappa):.5f}',
             horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.show()
