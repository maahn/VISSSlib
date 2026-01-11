import numpy as np
import VISSSlib

nSample = 100
seed = 0


class TestVolume(object):
    def test_VolumeInterpolation(self):
        width = 1280
        height = 1024

        maxSharpnessSizes = tuple()
        maxSharpnessLeader = tuple()
        maxSharpnessFollower = tuple()
        correctForSmallOnes = False

        np.random.seed(seed)
        phi, theta, Of_z = np.random.random(3)
        minDmax, maxDmax = 0, 20
        sizeBins = tuple(np.linspace(minDmax, maxDmax))
        D_highRes, V_highRes = VISSSlib.distributions._estimateVolumes(
            width,
            height,
            correctForSmallOnes,
            phi,
            theta,
            Of_z,
            sizeBins,
            maxSharpnessSizes,
            maxSharpnessLeader,
            maxSharpnessFollower,
            nSteps=21,
            interpolate=True,
        )
        D_lowRes, V_lowRes = VISSSlib.distributions._estimateVolumes(
            width,
            height,
            correctForSmallOnes,
            phi,
            theta,
            Of_z,
            sizeBins,
            maxSharpnessSizes,
            maxSharpnessLeader,
            maxSharpnessFollower,
            nSteps=2,
            interpolate=True,
        )

        assert np.all(np.abs(V_lowRes - V_highRes) / V_highRes < 1e-2)

    def test_volumeEstimate(self):
        width = 1280
        height = 1024

        # no rotation!
        phi, theta, Of_z = 0, 0, 0

        V = VISSSlib.distributions._estimateVolume(width, height, phi, theta, Of_z)

        assert np.isclose(V, width * width * height)
