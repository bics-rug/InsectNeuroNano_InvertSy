"""
Package the contains the default agents.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

import numbers
from abc import ABC

from ._helpers import eps, RNG
from ..__helpers import __data__

from invertpy.sense import PolarisationSensor, CompoundEye, Antennas, Sensor
from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertpy.brain import WillshawNetwork, PolarisationCompass, Component
from invertpy.brain.centralcomplex import StoneCX, VectorMemoryCX, CentralComplexBase
from invertpy.brain.centralcomplex.minimal_device import MinimalDeviceCX
from invertpy.brain.centralcomplex.familiarity import FamiliarityIntegratorCX, FamiliarityCX
from invertpy.brain.centralcomplex.vectormemory import mem2vector
from invertpy.brain.mushroombody import VectorMemoryMB, IncentiveCircuit, VisualIncentiveCircuit
from invertpy.brain.memory import MemoryComponent
from invertpy.brain.activation import winner_takes_all, relu
from invertpy.brain.compass import ring2sph
from invertpy.brain.preprocessing import Whitening, ZernikeMoments, MentalRotation, LateralInhibition, pca

from scipy.spatial.transform import Rotation as R
from copy import copy

import loguru as lg
import numpy as np

import os

__data_calibrate__ = os.path.join(__data__, "calibration")
DEFAULT_ACC = 0.15
DEFAULT_DRAG = 0.15


class Agent(object):
    def __init__(self, xyz=None, ori=None, speed=0.1, delta_time=1., dtype='float32', name='agent', rng=RNG, noise=0.,
                 **kwargs):
        """
        Abstract agent class that holds all the basic methods and attributes of an agent such as:

        - 3D position and initial position
        - 3D orientation and initial orientation
        - delta time: how fast its internal clock is ticking
        - delta x: how fast is it moving
        - name
        - sensors
        - brain components
        - translation and rotation methods

        Parameters
        ----------
        xyz: np.ndarray[float], optional
            the initial 3D position of the agent. Default is p=[0, 0, 0]
        ori: R, optional
            the initial 3D orientation of the agent. Default is q=[1, 0, 0, 0]
        speed: float, optional
            the agent's speed. Default is dx=0.1 meters/sec
        delta_time: float, optional
            the agent's internal clock speed. Default is 1 tick/second
        name: str, optional
            the name of the agent. Default is 'agent'
        dtype: np.dtype, optional
            the type of the agents parameters
        """
        if xyz is None:
            xyz = [0, 0, 0]
        if ori is None:
            ori = R.from_euler('Z', 0)

        if not hasattr(self, "_sensors"):
            self._sensors = []  # type: list[Sensor]
        if not hasattr(self, "_brain"):
            self._brain = []  # type: list[Component]

        if not hasattr(self, "_xyz"):
            self._xyz = np.array(xyz, dtype=dtype)
        if not hasattr(self, "_ori"):
            self._ori = ori

        if not hasattr(self, "_xyz_init"):
            self._xyz_init = self._xyz.copy()
        if not hasattr(self, "_ori_init"):
            self._ori_init = copy(self._ori)

        self._dt_default = delta_time  # seconds
        self._dx = speed  # meters / second
        self._velocity = np.zeros(2, dtype=float)
        self._acceleration = DEFAULT_ACC
        self._drag = DEFAULT_DRAG

        self.name = name
        self.dtype = dtype

        self.rng = rng
        self._noise = noise

    def reset(self):
        """
        Re-initialises the parameters, sensors and brain components of the agent.
        """
        xyz_c = copy(self._xyz)
        ori_c = copy(self._ori)
        self._xyz = copy(self._xyz_init)
        self._ori = copy(self._ori_init)
        self._velocity = np.zeros(2, dtype=float)

        for sensor in self.sensors:
            # reset the sensor (this does not reset the sensor's centre of mass)
            sensor.reset()

            # reset the position and orientation of the sensors separately with respect to the agent's new position
            sensor.translate(self._xyz_init - xyz_c)
            sensor.rotate(ori_c.inv() * self._ori_init)

        for component in self.brain:
            component.reset()

    def _sense(self, *args, **kwargs):
        """
        Senses the environment. This method needs to be implemented by the sub-class.

        Returns
        -------
        out
            the output of the sensors

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def _act(self):
        """
        Acts in the environment. This method needs to be implemented by the sub-class.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """
        Senses the environment and then acts in it given the parameters.

        Returns
        -------
        out
            the output of the sensors
        """
        act = kwargs.pop('act', True)
        callback = kwargs.pop('callback', None)

        out = self._sense(*args, **kwargs)
        if act:
            self._act()

        if callback is not None:
            callback(self)

        return out

    def __repr__(self):
        return ("Agent(xyz=[%.2f, %.2f, %.2f] m, ori=[%.0f, %.0f, %.0f] degrees, speed=%.2f m/s, "
                "#sensors=%d, #brain_components=%d, name='%s')") % (
            self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self._dx,
            len(self.sensors), len(self.brain), self.name
        )

    def move_forward(self, dx=None, dt=None):
        """
        Move towards the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([1, 0, 0], dx, dt)

    def move_backward(self, dx=None, dt=None):
        """
        Move towards the opposite of the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([-1, 0, 0], dx, dt)

    def move_right(self, dx=None, dt=None):
        """
        Move sideways to the right of the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([0, 1, 0], dx, dt)

    def move_left(self, dx=None, dt=None):
        """
        Move sideways to the left of the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([0, -1, 0], dx, dt)

    def move_towards(self, direction_xyz, dx=None, dt=None):
        """
        Moves the agent towards a 3D direction (locally - relative to the current direction) using for a dx/dt distance.

        Parameters
        ----------
        direction_xyz: np.ndarray[float], list[float]
            3D vector showing the direction of motion
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        if dt is None:
            dt = self._dt_default
        if dx is None:
            dx = self._dx

        # compute the step size based on the new delta time
        dx = dx * dt

        self.translate(self._ori.apply(dx * np.array(direction_xyz)))

    def rotate(self, d_ori: R):
        """
        Rotate the agent and its sensor on the spot.

        Parameters
        ----------
        d_ori: Rotation
            the rotation to apply on the current direction of the agent
        """
        self._ori = self._ori * d_ori
        for sensor in self._sensors:
            sensor.rotate(d_ori, around_xyz=self._xyz)

    def translate(self, d_xyz):
        """
        Translates the agent and its sensors by adding the given vector in global coordinates.

        Parameters
        ----------
        d_xyz: np.ndarray[float], list[float]
            the vector to add in global coordinates
        """
        self._xyz += np.array(d_xyz, dtype=self.dtype)
        for sensor in self._sensors:
            sensor.translate(d_xyz)

    def add_sensor(self, sensor, local=False):
        """
        Adds a sensor to the agent. By default, the sensor is assumed to have its orientation and position in global
        coordinates, but this can be changed through the 'local' option.

        Parameters
        ----------
        sensor: Sensor
            The sensor to add.
        local: bool
            If True, then the orientation and coordinates of the sensor are supposed to be local (with respect to the
            agent's orientation and coordinates, otherwise it is global. Default is False (global).
        """
        if local:
            sensor.rotate(self._ori)
            sensor.translate(self._xyz)
        self._sensors.append(sensor)

    def add_brain_component(self, component: Component):
        self._brain.append(component)

    @property
    def sensors(self):
        """
        The sensors of the agent.

        Returns
        -------
        sensors: list[Sensor]
        """
        return self._sensors

    @property
    def brain(self):
        """
        The brain components of the agent.

        Returns
        -------
        brain: list[Component]
        """
        return self._brain

    @property
    def xyz(self):
        """
        The position of the agent.

        Returns
        -------
        xyz: np.ndarray[float]

        See Also
        --------
        Agent.position
        """
        return self._xyz

    @xyz.setter
    def xyz(self, v):
        """
        The position of the agent.

        Parameters
        ----------
        v: np.ndarray[float]

        See Also
        --------
        Agent.position
        """
        self.translate(np.array(v, dtype=self.dtype) - self._xyz)

    @property
    def x(self):
        """
        The x component of the position of the agent.

        Returns
        -------
        x: float
        """
        return self._xyz[0]

    @property
    def y(self):
        """
        The y component of the position of the agent.

        Returns
        -------
        y: float
        """
        return self._xyz[1]

    @property
    def z(self):
        """
        The z component of the position of the agent.

        Returns
        -------
        z: float
        """
        return self._xyz[2]

    @property
    def ori(self):
        """
        The orientation of the agent

        Returns
        -------
        ori: R

        See Also
        --------
        Agent.orientation
        """
        return self._ori

    @ori.setter
    def ori(self, v):
        """
        Parameters
        ----------
        v: R

        See Also
        --------
        Agent.orientation
        """
        self.rotate(d_ori=self._ori.inv() * v)

    @property
    def euler(self):
        """
        The orientation of the agent as euler angles (yaw, pitch, roll) in radians.

        Returns
        -------
        euler: np.ndarray[float]
        """
        return self._ori.as_euler('ZYX', degrees=False)

    @property
    def yaw(self):
        """
        The yaw of the agent in radians.

        Returns
        -------
        yaw: float
        """
        return self.euler[0]

    @property
    def pitch(self):
        """
        The pitch of the agent in radians.

        Returns
        -------
        pitch: float
        """
        return self.euler[1]

    @property
    def roll(self):
        """
        The roll of the agent in radians.

        Returns
        -------
        roll: float
        """
        return self.euler[2]

    @property
    def euler_deg(self):
        """
        The orientation of the agent as euler angles (yaw, pitch, roll) in degrees.

        Returns
        -------
        euler_deg: np.ndarray[float]
        """
        return self._ori.as_euler('ZYX', degrees=True)

    @property
    def yaw_deg(self):
        """
        The yaw of the agent in degrees.

        Returns
        -------
        yaw_deg: float
        """
        return self.euler_deg[0]

    @property
    def pitch_deg(self):
        """
        The pitch of the agent in degrees.

        Returns
        -------
        pitch_deg: float
        """
        return self.euler_deg[1]

    @property
    def roll_deg(self):
        """
        The roll of the agent in degrees.

        Returns
        -------
        roll_deg: float
        """
        return self.euler_deg[2]

    @property
    def position(self):
        """
        The position of the agent.

        Returns
        -------
        position: np.ndarray[float]

        See Also
        --------
        Agent.xyz
        """
        return self._xyz

    @property
    def orientation(self):
        """
        The orientation of the agent.

        Returns
        -------
        orientation: np.ndarray[float]

        See Also
        --------
        Agent.ori
        """
        return self._ori

    @property
    def step_size(self):
        """
        The step size (dx) per delta time (dt).

        Returns
        -------
        dx: float
        """
        return self._dx

    @step_size.setter
    def step_size(self, value):
        self._dx = value

    @property
    def delta_time(self):
        """
        The delta time (dt) among time-steps.

        Returns
        -------
        dt: float
        """
        return self._dt_default


class VisualProcessingAgent(Agent, ABC):

    def __init__(self, eye=None, saturation=7., nb_scans=1, nb_visual=None,
                 lateral_inhibition=False, mental_scanning=0, zernike=False, whitening=pca,
                 *args, **kwargs):
        """
        Agent specialised in the visual navigation task. It contains the CompoundEye as a sensor and the mushroom body
        as the brain component.

        Parameters
        ----------
        eye: CompoundEye, optional
            instance of the compound eye of the agent. Default is a compound eye with 5000 ommatidia, with 15 deg
            acceptance angle each, sensitive to green only and not sensitive to polarised light.
        memory: MemoryComponent, optional
            instance of a mushroom body model as a processing component. Default is the WillshawNetwork with #PN equal
            to the number of ommatidia, #KC equal to 40 x #PN, sparseness is 1%, and eligibility trace (lambda) is 0.1
        saturation: float, optional
            the maximum radiation level that the eye can handle, anything above this threshold will be saturated.
            Default is 1.5
        nb_scans: int, optional
            the number of scans during the route following task. Default is 7
        zernike: bool, optional
            whether to transform the visual input into the frequency domain by using the DCT method. Default is False
        """
        Agent.__init__(self, *args, **kwargs)

        if eye is None:
            eye = CompoundEye(nb_input=1000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10), omm_res=saturation,
                              c_sensitive=[0, 0., 1., 0., 0.])

        nb_input = eye.nb_ommatidia
        if nb_visual is None:
            nb_visual = nb_input

        self.add_sensor(eye)

        self._eye = eye  # type: CompoundEye

        self._pref_angles = None
        """
        The preferred angles for scanning
        """

        if nb_scans <= 1 and mental_scanning <= 1:
            self._pref_angles = np.array([0], dtype=self.dtype)
        elif nb_scans > 1:
            self._pref_angles = np.linspace(-60, 60, nb_scans)
        elif mental_scanning > 1:
            self._pref_angles = np.linspace(0, 360, mental_scanning, endpoint=False)

        self._preprocessing = []
        """
        List of the preprocessing components
        """

        if lateral_inhibition:
            self._preprocessing.append(LateralInhibition(ori=eye.omm_ori, nb_neighbours=6))

        if mental_scanning > 1:
            self._preprocessing.append(MentalRotation(eye=eye, pref_angles=np.deg2rad(self._pref_angles)))

        if whitening is not None:
            self._preprocessing.append(Whitening(nb_input=nb_input, nb_output=nb_visual,
                                                 w_method=whitening, dtype=eye.dtype))
            nb_input = nb_visual
        if zernike:
            self._preprocessing.append(ZernikeMoments(nb_input=nb_input, ori=eye.omm_ori, dtype=eye.dtype))

        if self.__class__ == VisualProcessingAgent:
            self.reset()

    def reset(self):
        super().reset()

        for process in self._preprocessing:
            process.reset()

    def _sense(self, sky=None, scene=None, omm_responses=None):
        return self.get_pn_responses(sky=sky, scene=scene, omm_responses=omm_responses)

    def calibrate(self, sky=None, scene=None, omm_responses=None, nb_samples=None, radius=2.):
        """
        Approximates the calibration of the optic lobes of the agent.
        In this case, it randomly collects a number of samples (in different positions and direction) in a radius
        around the nest. These samples are used in order to build a PCA whitening map, that transforms the visual
        input from the ommatidia to a white signal thying to maximise its information.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        omm_responses: np.ndarray[float]
            the pre-rendered ommatidia responses. Default is None
        nb_samples: int, optional
            the number of samples to use. Default is the number of ommatidia
        radius: float, optional
            the radius around the nest from where the samples will be taken

        Returns
        -------
        xyz: list[np.ndarray[float]]
            the positions of the agent for every sample
        ori: list[R]
            the orientations of the agent for every sample
        """
        xyz = copy(self.xyz)
        ori = copy(self.ori)

        if omm_responses is not None:
            nb_samples = omm_responses.shape[0]

        li = ""
        for proc in self.preprocessing:
            if isinstance(proc, LateralInhibition):
                li = "_li"

        cal_file = os.path.join(__data_calibrate__, f"calibration{li}_{nb_samples}.npz")
        if os.path.exists(cal_file):
            data = np.load(cal_file, allow_pickle=True)

            lg.logger.debug(f"Calibration data loaded from: '{cal_file}'")

            samples = data["samples"]
            xyzs = data["xyz"]
            oris = data["ori"]
        else:
            nb_ms = self.nb_mental_rotations

            samples = np.zeros((nb_samples * nb_ms, self._eye.nb_ommatidia), dtype=self.dtype)
            xyzs, oris = [], []
            for i in range(nb_samples):
                if omm_responses is None:
                    self.xyz = xyz + self.rng.uniform(-radius, radius, 3) * np.array([1., 1., 0])
                    self.ori = R.from_euler("Z", self.rng.uniform(-180, 180), degrees=True)
                    xyzs.append(copy(self.xyz))
                    oris.append(copy(self.ori))

                lg.logger.debug("Calibration: %d/%d - x: %.2f, y: %.2f, z: %.2f, yaw: %d" % (
                    i + 1, nb_samples, self.x, self.y, self.z, self.yaw_deg))
                samples[i*nb_ms:(i+1)*nb_ms] = self.get_pn_responses(
                    sky, scene, omm_responses[i] if omm_responses is not None else None, pre_whitened=True)

            np.savez(cal_file, samples=samples, xyz=xyzs, ori=oris)
            lg.logger.info(f"Calibration data saved in: '{cal_file}'")

        for p in self._preprocessing:
            if isinstance(p, Whitening):
                p.reset(samples)
                lg.logger.info("Calibration: DONE!")

        self.xyz = xyz
        self.ori = ori

        return xyzs, oris

    def get_pn_responses(self, sky=None, scene=None, omm_responses=None, pre_whitened=False, raw=False):
        """
        Transforms the current snapshot of the environment into the PN responses.

        - Apply DCT (if applicable)
        - Apply PCA whitening (if applicable)

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        omm_responses: np.ndarray[float]
            the pre-rendered ommatidia responses. Default is None
        pre_whitened: bool
        raw : bool

        Returns
        -------
        r: np.ndarray[float]
            the responses of the PNs
        """
        if omm_responses is None:  # if rendered ommatidia responses are provided, use them
            omm_responses = self._eye(sky=sky, scene=scene)

        r = np.clip(omm_responses.mean(axis=1), 0, 1)
        if raw:
            pass
        elif pre_whitened:
            i = 0
            while not isinstance(self._preprocessing[i], Whitening) and i < len(self._preprocessing):
                r = self._preprocessing[i](r)
                i += 1
        else:
            for process in self._preprocessing:
                r = process(r)

        if r.ndim < 2:
            r = r[np.newaxis, :]

        if self.is_calibrated and False:
            import matplotlib.pyplot as plt

            plt.figure("preprocessing", figsize=(10, 10))

            yaw, pitch, _ = self._eye.omm_ori.as_euler('ZYX', degrees=True).T
            for i in range(r.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.scatter(yaw, -pitch, s=20, c=r[i], cmap='coolwarm', vmin=-1, vmax=1)

            plt.show()
        return r

    def get_omm_responses(self, sky=None, scene=None):
        """
        Generates the current snapshot of the environement.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None

        Returns
        -------
        r: np.ndarray[float]
            the responses of the ommatidia
        """
        return np.clip(self._eye(sky=sky, scene=scene).mean(axis=1), 0, 1)

    @property
    def nb_mental_rotations(self):
        return len(self._pref_angles)

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def pref_angles(self):
        """
        The preferred angles of the agent, where it will look at when scanning
        """
        return self._pref_angles

    @property
    def nb_scans(self):
        """
        The number of scans to be applied
        """
        return self._pref_angles.shape[0]

    @property
    def nb_visual_cues(self):
        return self.preprocessing[-1].nb_output

    @property
    def is_calibrated(self):
        """
        Indicates if calibration has been completed
        """
        for p in self._preprocessing:
            if isinstance(p, Whitening):
                return p.calibrated
        return False

    @property
    def eye(self):
        return self._eye


class CentralComplexAgent(Agent, ABC):
    def __init__(self, cx_class=StoneCX, cx_params=None, *args, **kwargs):
        Agent.__init__(self, *args, **kwargs)

        pol_sensor = PolarisationSensor(nb_input=60, field_of_view=56, degrees=True, noise=self._noise, rng=self.rng)
        pol_compass = PolarisationCompass(nb_pol=60, loc_ori=copy(pol_sensor.omm_ori), nb_sol=8, integrated=True,
                                          noise=self._noise, rng=self.rng)
        if cx_params is None:
            cx_params = {}
        cx_params.setdefault('nb_tb1', 8)
        cx_params.setdefault('noise', self._noise)
        cx_params.setdefault('rng', self.rng)

        cx = cx_class(**cx_params)

        self.add_sensor(pol_sensor, local=True)
        self.add_brain_component(pol_compass)
        self.add_brain_component(cx)

        self._pol_sensor = pol_sensor
        self._pol_compass = pol_compass
        self._cx = cx
        self._p_phi = None

        self._default_flow = self._dx * np.ones(2) / np.sqrt(2)

    def _sense(self, sky=None, scene=None, flow=None, **kwargs):
        """
        Using its only sensor (the dorsal rim area) it senses the radiation from the sky which is interrupted by the
        given scene, and the optic flow for self motion calculation.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        flow: np.ndarray[float], optional
            the optic flow. Default is the preset optic flow

        Returns
        -------
        out: np.ndarray[float]
            the output of the central complex
        """
        if sky is None:
            r = 0.
        else:
            r = self.pol_sensor(sky=sky, scene=scene)

        r_tcl = self.compass(r_pol=r, ori=self._pol_sensor.ori)
        _, phi = ring2sph(r_tcl)
        phi = self.yaw

        if flow is None:
            if self._p_phi is None:
                flow = self._default_flow
            else:
                d_phi = (self.yaw - self._p_phi + np.pi) % (2 * np.pi) - np.pi
                v = self._dx * np.exp(1j * d_phi)
                flow = self._cx.get_flow(-d_phi, np.array([v.imag, v.real]), 0)

            self._p_phi = self.yaw

            # if scene is None:
            #     flow = self._default_flow
            # else:
            #     flow = optic_flow(world, self._dx)

        return self._cx(phi=phi, flow=flow, **kwargs)

    def _act(self):
        """
        Uses the output of the central complex to compute the next movement and moves the agent to its new position.
        """
        steer = self.get_steering(self._cx) * 0.25  # to kill the noise a bit!
        yaw_pre = self.yaw
        self.rotate(R.from_euler('Z', steer, degrees=False))

        thrust = np.array([np.cos(self.yaw - yaw_pre), np.sin(self.yaw - yaw_pre)]) * self._acceleration * self.delta_time
        x, y = (self._velocity + thrust) * (1 - self._drag) ** self.delta_time
        self._velocity[:] = [x, y]
        self.move_towards([x, y, 0])

    @property
    def central_complex(self):
        """
        The central complex model of the agent.

        Returns
        -------
        CentralComplexBase
        """
        return self._cx

    @property
    def pol_sensor(self):
        """
        The Polarisation sensor of the agent.

        Returns
        -------
        PolarisationSensor
        """
        return self._pol_sensor

    @property
    def compass(self):
        """
        The POL compass model of the agent.

        Returns
        -------
        CelestialCompass
        """
        return self._pol_compass

    @staticmethod
    def get_steering(cx):
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        cx: CentralComplexBase
            the central complex instance of the agent

        Returns
        -------
        output: float
            the angle of steering in radians
        """

        motor = cx.r_motor
        motor += 1. * (cx.rng.rand() - 0.5)

        output = motor[0] - motor[1]
        return output

class MinimalDeviceCentralComplexAgent(Agent, ABC):
    def __init__(self, cx_class=MinimalDeviceCX, cx_params=None, *args, **kwargs):
        Agent.__init__(self, *args, **kwargs)

        pol_sensor = MinimalDevicePolarisationSensor(POL_method="single_0", nb_lenses=6, omm_photoreceptor_angle=2, field_of_view=56, degrees=True)
        #pol_compass = PolarisationCompass(nb_pol=60, loc_ori=copy(pol_sensor.omm_ori), nb_sol=8, integrated=True,
        #                      noise, rng=self.rng)

        if cx_params is None:
            cx_params = {}

        cx = cx_class(**cx_params)

        self.add_sensor(pol_sensor, local=True)
        self.add_brain_component(cx)

        self._pol_sensor = pol_sensor
        self._cx = cx
        self._p_phi = None

        self._default_flow = self._dx * np.ones(2) / np.sqrt(2)

    def _sense(self, sky=None, scene=None, flow=None, **kwargs):
        """
        Using its only sensor (the dorsal rim area) it senses the radiation from the sky which is interrupted by the
        given scene, and the optic flow for self motion calculation.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        flow: np.ndarray[float], optional
            the optic flow. Default is the preset optic flow

        Returns
        -------
        out: np.ndarray[float]
            the output of the central complex
        """
        if sky is None:
            r = 0.
        else:
            r = self.pol_sensor(sky=sky)

        output = self._cx(POL_direction=r, **kwargs)
        return output

    def _act(self):
        """
        Uses the output of the central complex to compute the next movement and moves the agent to its new position.
        """
        steer = self.get_steering(self._cx) * 0.25  # to kill the noise a bit!
        print(f"{steer=}")
        yaw_pre = self.yaw
        self.rotate(R.from_euler('Z', steer, degrees=False))

        thrust = np.array([np.cos(self.yaw - yaw_pre), np.sin(self.yaw - yaw_pre)]) * self._acceleration * self.delta_time
        x, y = (self._velocity + thrust) * (1 - self._drag) ** self.delta_time
        self._velocity[:] = [x, y]
        self.move_towards([x, y, 0])

    @property
    def central_complex(self):
        """
        The central complex model of the agent.

        Returns
        -------
        CentralComplexBase
        """
        return self._cx

    @property
    def pol_sensor(self):
        """
        The Polarisation sensor of the agent.

        Returns
        -------
        PolarisationSensor
        """
        return self._pol_sensor

    @staticmethod
    def get_steering(cx):
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        cx: CentralComplexBase
            the central complex instance of the agent

        Returns
        -------
        output: float
            the angle of steering in radians
        """
        motor = cx.r_motor * 1e+08
        motor += 1. * (np.random.rand() - 0.5)

        output = motor[1] - motor[0]
        return output

class PathIntegrationAgent(CentralComplexAgent):
    pass


class VectorMemoryAgent(CentralComplexAgent):

    def __init__(self, nb_feeders=1, *args, **kwargs):
        """
        Agent specialised in the path integration task. It contains the Dorsal Rim Area as a sensor, the polarised
        light compass and the central complex as brain components.
        """

        cx_params = {
            "nb_mbon": 6,
            "nb_vectors": nb_feeders+1
        }
        super().__init__(cx=VectorMemoryCX, cx_params=cx_params, *args, **kwargs)

        # cx = FamiliarityCX(nb_tb1=8, nb_mbon=6, noise=self._noise, rng=self.rng)

    def _sense(self, mbon=None, vec=None, **kwargs):
        """
        Using its only sensor (the dorsal rim area) it senses the radiation from the sky which is interrupted by the
        given scene, and the optic flow for self motion calculation.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        flow: np.ndarray[float], optional
            the optic flow. Default is the preset optic flow

        Returns
        -------
        out: np.ndarray[float]
            the output of the central complex
        """

        if mbon is None:
            mbon = np.array([0] * self.central_complex.nb_mbon)
        return super()._sense(mbon=mbon, vec=vec, **kwargs)

    @property
    def central_complex(self):
        """
        The central complex model of the agent.

        Returns
        -------
        VectorMemoryCX
        """
        return self._cx


class RouteFollowingAgent(VisualProcessingAgent, CentralComplexAgent):

    def __init__(self, mb_class=None, mb_params=None, *args, **kwargs):

        VisualProcessingAgent.__init__(self, *args, **kwargs)

        if mb_class is None:
            mb_class = WillshawNetwork
        if mb_params is None:
            mb_params = {}
        mb_params.setdefault("nb_input", self.preprocessing[-1].nb_output)
        mb_params.setdefault("nb_sparse", 40 * mb_params["nb_input"])
        mb_params.setdefault("sparseness", 10 / mb_params["nb_sparse"])
        mb_params.setdefault("eligibility_trace", 0.)

        self._mem = mb_class(**mb_params)

        cx_params = {
            "nb_mbon": self._mem.nb_output,
            "gain": .5
        }
        # vis_sensors = self.sensors
        CentralComplexAgent.__init__(self, cx_class=FamiliarityCX, cx_params=cx_params, *args, **kwargs)
        # CentralComplexAgent.__init__(self, cx_class=FamiliarityIntegratorCX, cx_params=cx_params, *args, **kwargs)
        # self._sensors = vis_sensors + self.sensors

        self._mem.reset()

        if self.__class__ == RouteFollowingAgent:
            self.reset()

    def reset(self):
        CentralComplexAgent.reset(self)
        VisualProcessingAgent.reset(self)

        # self._mem.reset()

    def _sense(self, sky=None, scene=None, flow=None, **kwargs):
        omm_responses = kwargs.pop("omm_responses", None)
        r_pn = VisualProcessingAgent._sense(self, sky=sky, scene=scene, omm_responses=omm_responses)
        r_mbon = self._mem(cs=r_pn, us=np.array([self.update, 0], dtype=self.dtype))

        return CentralComplexAgent._sense(self, sky=sky, scene=scene, flow=flow, mbon=r_mbon, **kwargs)

    @property
    def mushroom_body(self):
        return self._mem

    @property
    def update(self):
        """
        Whether the memory will be updated or not
        """
        return self._mem.update

    @update.setter
    def update(self, v):
        """
        Enables (True) or disables (False) the memory updates.

        Parameters
        ----------
        v: bool
            memory updates
        """
        self._mem.update = v


class VisualNavigationAgent(VisualProcessingAgent):

    def __init__(self, memory=None, *args, **kwargs):
        """
        Agent specialised in the visual navigation task. It contains the CompoundEye as a sensor and the mushroom body
        as the brain component.

        Parameters
        ----------
        eye: CompoundEye, optional
            instance of the compound eye of the agent. Default is a compound eye with 5000 ommatidia, with 15 deg
            acceptance angle each, sensitive to green only and not sensitive to polarised light.
        memory: MemoryComponent, optional
            instance of a mushroom body model as a processing component. Default is the WillshawNetwork with #PN equal
            to the number of ommatidia, #KC equal to 40 x #PN, sparseness is 1%, and eligibility trace (lambda) is 0.1
        saturation: float, optional
            the maximum radiation level that the eye can handle, anything above this threshold will be saturated.
            Default is 1.5
        nb_scans: int, optional
            the number of scans during the route following task. Default is 7
        zernike: bool, optional
            whether to transform the visual input into the frequency domain by using the DCT method. Default is False
        """
        super().__init__(*args, **kwargs)

        if memory is None:
            nb_visual = self.preprocessing[-1].nb_output
            # #KC = 40 * #PN
            memory = WillshawNetwork(nb_input=nb_visual, nb_sparse=4000, sparseness=0.01, eligibility_trace=.1)
        self.add_brain_component(memory)

        self._mem = memory  # type: MemoryComponent

        self._familiarity = np.zeros_like(self._pref_angles)
        """
        The familiarity of each preferred angle
        """

        self.reset()

    def reset(self):
        super().reset()

        self._familiarity = np.zeros_like(self._pref_angles)

    def _sense(self, sky=None, scene=None, omm_responses=None, **kwargs):
        """
        Using its only sensor (the compound eye) it senses the radiation from the sky and the given scene to calculate
        the familiarity. In the case of route following (when there is no update), it scans in all the preferred angles
        and calculates the familiarity in all of them.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        omm_responses: np.ndarray[float]
            the pre-rendered ommatidia responses. Default is None

        Returns
        -------
        familiarity: np.ndarray[float]
            how familiar does the agent is with every scan made
        """

        self._familiarity = np.zeros_like(self._pref_angles)

        if self.update:
            r = self.get_pn_responses(sky=sky, scene=scene, omm_responses=omm_responses)
            # us = np.zeros_like(self.pref_angles)
            # us[0] = 1.
            us = np.cos(np.deg2rad(self.pref_angles))
            self._mem(cs=r, us=us)
            if isinstance(self._mem, VisualIncentiveCircuit):
                fam = np.mean(self._mem.familiarity, axis=-1)
            else:
                fam = self._mem.familiarity
            if isinstance(fam, numbers.Number):
                fam_raw = np.array([fam])
            else:
                fam_raw = fam
            if self._mem.ndim > 1 or self._mem.nb_output > 1:
                self._familiarity[:] = fam_raw.flatten()
            else:
                self._familiarity[0] = fam_raw[0].flatten()
        else:
            inp, hid, out = [], [], []

            if self._mem.ndim > 1 or self._mem.nb_output > 1:

                r = self.get_pn_responses(sky=sky, scene=scene, omm_responses=omm_responses)
                self._mem(cs=r)
                if isinstance(self._mem, VisualIncentiveCircuit):
                    fam = np.mean(self._mem.familiarity, axis=-1)
                else:
                    fam = self._mem.familiarity
                self._familiarity[:] = fam.flatten()

                inp.extend(self._mem.r_inp)
                hid.extend(self._mem.r_hid)
                out.extend(self._mem.r_out)
            else:
                ori = copy(self.ori)

                r_inp = copy(self._mem.r_inp)
                r_hid = copy(self._mem.r_hid)
                r_out = copy(self._mem.r_out)

                for i, angle in enumerate(self._pref_angles):
                    self._mem._inp = copy(r_inp)
                    self._mem._hid = copy(r_hid)
                    self._mem._out = copy(r_out)

                    self.ori = ori * R.from_euler('Z', angle, degrees=True)
                    r = self.get_pn_responses(sky=sky, scene=scene, omm_responses=omm_responses)
                    self._mem(cs=r)
                    if isinstance(self._mem, VisualIncentiveCircuit):
                        fam = np.mean(self._mem.familiarity[0], axis=-1)
                    else:
                        fam = self._mem.familiarity
                    self._familiarity[i] = fam[0].flatten()

                    inp.append(copy(self._mem.r_inp))
                    hid.append(copy(self._mem.r_hid))
                    out.append(copy(self._mem.r_out))

                self.ori = ori

                i = self._familiarity.argmax()

                self._mem._inp = inp[i]
                self._mem._hid = hid[i]
                self._mem._out = out[i]

        return self._familiarity

    def _act(self):
        """
        Uses the familiarity vector to compute the next movement and moves the agent to its new position.
        """
        steer = self.get_steering(self.familiarity, self.pref_angles, max_steering=20, degrees=True)
        self.rotate(R.from_euler('Z', steer, degrees=True))
        self.move_forward()

    @property
    def familiarity(self):
        """
        The familiarity of the latest snapshot per preferred angle
        """
        return self._familiarity

    @property
    def update(self):
        """
        Whether the memory will be updated or not
        """
        return self._mem.update

    @update.setter
    def update(self, v):
        """
        Enables (True) or disables (False) the memory updates.

        Parameters
        ----------
        v: bool
            memory updates
        """
        self._mem.update = v

    @property
    def memory_component(self):
        return self._mem

    @staticmethod
    def get_steering_from_mbons(r_mbon, gain=1., max_steering=None, degrees=True):
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        familiarity: np.ndarray[float]
            the familiarity vector computed by scanning the environment
        pref_angles: np.ndarray[float]
            the preference angle associated to the values of the familiarity vector
        max_steering: float, optional
            the maximum steering allowed for the agent. Default is 30 degrees
        degrees: bool, optional
            whether the max_steering is in degrees or radians. Default is degrees

        Returns
        -------
        output: float
            the angle of steering in radians
        """
        if degrees:
            angle = lambda x: x
        else:
            angle = np.rad2deg
        if max_steering is None:
            max_steering = 5.
        else:
            max_steering = angle(max_steering)

        if gain is None:
            gain = max_steering

        if r_mbon.size < 2:
            step = 1
        elif r_mbon.size == 2 or r_mbon.size == 6:
            step = 2
        else:
            step = r_mbon.size // 2
        steering = (r_mbon[..., 1::step] - r_mbon[..., 0::step]).mean()

        if np.isnan(steering):
            steering = 0.

        # lg.logger.debug("Steering: %.2f" % steering, end=", ")
        steering = np.clip(gain * steering, -max_steering, max_steering)
        # lg.logger.debug("Clipped: %.2f" % steering)
        if not degrees:
            steering = np.deg2rad(steering)

        return steering

    @staticmethod
    def get_steering(familiarity, pref_angles, max_steering=None, degrees=True):
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        familiarity: np.ndarray[float]
            the familiarity vector computed by scanning the environment
        pref_angles: np.ndarray[float]
            the preference angle associated to the values of the familiarity vector
        max_steering: float, optional
            the maximum steering allowed for the agent. Default is 30 degrees
        degrees: bool, optional
            whether the max_steering is in degrees or radians. Default is degrees

        Returns
        -------
        output: float
            the angle of steering in radians
        """
        if max_steering is None:
            max_steering = np.deg2rad(30)
        elif degrees:
            max_steering = np.deg2rad(max_steering)
        if degrees:
            pref_angles = np.deg2rad(pref_angles)
        # r = familiarity - familiarity.min()
        # r = np.power(1e-02, np.absolute(1 - familiarity))
        # r = familiarity / (familiarity.sum() + eps)
        r = familiarity
        pref_angles_c = r * np.exp(1j * pref_angles)

        steer_vector = np.sum(pref_angles_c)
        steer = (np.angle(steer_vector) + np.pi) % (2 * np.pi) - np.pi
        lg.logger.debug("Steering: %.2f" % np.rad2deg(steer), end=", ")
        # steer less for small vectors
        steer *= np.clip(np.absolute(steer_vector) * 2., eps, 1.)
        lg.logger.debug("Vector: %.2f" % np.absolute(steer_vector), end=", ")

        if np.isnan(steer):
            steer = 0.
        lg.logger.debug("Final steering: %.2f" % np.rad2deg(steer), end=", ")
        steer = np.clip(steer, -max_steering, max_steering)
        lg.logger.debug("Clipped: %d" % np.rad2deg(steer))
        if degrees:
            steer = np.rad2deg(steer)
        return steer


class NavigatingAgent(VectorMemoryAgent, VisualProcessingAgent):
    def __init__(self, nb_feeders=1, nb_odours=None, *args, **kwargs):
        VectorMemoryAgent.__init__(self, nb_feeders, *args, **kwargs)
        VisualProcessingAgent.__init__(self, *args, **kwargs)

        if nb_odours is None:
            # by default all feeders and the nest have different odours
            nb_odours = nb_feeders + 1  # +1 for the nest

        # odour IDs: 1 PN/odour
        nb_pn = 1 * nb_odours + self.nb_visual_cues
        nb_kc = 4000

        pol_sensor = PolarisationSensor(nb_input=60, field_of_view=56, degrees=True)
        pol_brain = PolarisationCompass(nb_pol=60, loc_ori=copy(self._pol_sensor.omm_ori), nb_sol=8, integrated=True)
        cx = VectorMemoryCX(nb_tb1=8, nb_mbon=6, nb_vectors=nb_odours, gain=0.1, noise=0.)

        antennas = Antennas(nb_tactile=0, nb_chemical=3, nb_chemical_dimensions=nb_odours)
        # mb = IncentiveCircuit(nb_cs=nb_pn, nb_us=2, nb_kc=nb_kc, ltm_charging_speed=0.1)  # IC

        mb = VectorMemoryMB(nb_cs=nb_pn, nb_us=2, nb_kc=nb_kc)  # vMB

        self.add_sensor(pol_sensor, local=True)
        self.add_sensor(antennas, local=True)
        self.add_brain_component(pol_brain)
        self.add_brain_component(cx)
        self.add_brain_component(mb)

        self._pol_sensor = pol_sensor
        self._antennas = antennas

        self._pol_brain = pol_brain
        self._cx = cx
        self._mb = mb

        # self._reinforcement_gamma = 0.997
        self._reinforcement_gamma = .5
        self._reinforcement = np.zeros(self._mb.nb_us, dtype=self.dtype)

        self._nb_feeders = nb_feeders
        self._nb_odours = nb_odours

    def _sense(self, sky=None, scene=None, odours=None, food=None, flow=None, reinforcement=None, **kwargs):

        r_vpn = VisualProcessingAgent.get_pn_responses(self, sky=sky, scene=scene)

        r_chem_pre = copy(self._antennas.responses[:, self._antennas.nb_tactile:]).mean(axis=0).reshape(
            self._antennas.nb_chemical, - 1).mean(axis=0)
        r_antenna = self._antennas(odours=odours)
        r_chem = r_antenna[:, self._antennas.nb_tactile:]

        # ignore the directional (left/right) signal
        r_chem = r_chem.mean(axis=0)
        # ignore the multiple sensors on the antenna
        r_chem = r_chem.reshape(self._antennas.nb_chemical, - 1).mean(axis=0)
        # # augment by using the gradient instead of the actual responses
        # r_chem_diff = np.r_[np.maximum(r_chem - r_chem_pre, 0), np.maximum(r_chem_pre - r_chem, 0)]
        # # normalise
        # r_chem_diff = r_chem_diff / np.maximum(r_chem_diff.sum(), eps)
        # # augment with actual odour identity
        # # r_chem_all = np.r_[r_chem, r_chem_diff]
        # # r_chem_all = r_chem_diff
        r_opn = r_chem

        if food is None:
            food = np.zeros(self.nb_odours, dtype=r_opn.dtype)

        # create PN responses that sum to 1
        # r_chem_all[:] = 0.  # erase odour information for testing the effect of motivation
        # r_pn = np.r_[r_chem_all, food] / (np.sum(r_chem_all) + np.sum(food) + eps)
        r_pn = np.r_[r_vpn.flatten(), r_opn.flatten()] / (np.sum(np.r_[r_vpn.flatten(), r_opn.flatten()]) + eps)

        self._reinforcement *= self._reinforcement_gamma
        if reinforcement is not None:
            self._reinforcement[:] = np.clip(self._reinforcement + reinforcement, 0, 1)

        # us = np.r_[self._reinforcement[:1], self._reinforcement[1:].max()]  # IC
        us = copy(self._reinforcement)  # vMB
        r_mbon = self._mb(cs=r_pn, us=us)

        return super(NavigatingAgent, self)._sense(sky=sky, scene=scene, flow=flow, mbon=r_mbon, **kwargs)

    @property
    def mushroom_body(self):
        return self._mb

    @property
    def antennas(self):
        return self._antennas

    @property
    def nb_vectors(self):
        return self._mb.nb_us - 1

    @property
    def nb_feeders(self):
        return self._nb_feeders

    @property
    def nb_odours(self):
        return self._nb_odours
