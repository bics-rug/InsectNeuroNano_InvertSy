import numbers

from invertsy.agent.agent import VisualNavigationAgent, PathIntegrationAgent

from invertpy.brain import MushroomBody
from invertpy.sense import CompoundEye, PolarisationSensor

import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.lines
import matplotlib.image
import numpy as np


def create_map_axis(world=None, nest=None, feeders=None, odour_spread=None, subplot=111, ax=None):
    """
    Draws a map with all the vegetation from the world (if given), the nest and feeder positions (if given) and returns
    the ongoing and previous paths of the agent, the agent's current position, the marker (arror) of the agents facing
    direction, the calibration points and the points where the agent is taken back on the route after replacing.

    Parameters
    ----------
    world: Seville2009, optional
        the world containing the vegetation. Default is None
    nest: np.ndarray[float], optional
        the position of the nest. Default is None
    feeders: list[float], list[np.ndarray[float]], np.ndarray[float]
        the position of the feeder. Default is None
    odour_spread: list[float]
        the spead of the different odours
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    line_c: matplotlib.lines.Line2D
        the ongoing path of the agent
    line_b: matplotlib.lines.Line2D
        the previous paths of the agent
    pos: matplotlib.collections.PathCollection
        the current position of the agent
    marker: tuple[np.ndarray[float], np.ndarray[float]]
        the marker parameters
    cal: matplotlib.collections.PathCollection
        the points on the map where the calibration took place
    poi: matplotlib.collections.PathCollection
        the points on the map where the agent was put back on the route
    """

    if ax is None:
        ax = plt.subplot(subplot)

    line_b, = ax.plot([], [], 'grey', lw=2)

    if nest is not None and odour_spread is not None:
        xy = odour_spread[0] * np.exp(1j * np.linspace(0, 2 * np.pi, 101, endpoint=True))
        ax.plot(xy.real + nest[1], xy.imag + nest[0], 'C0--', lw=2, alpha=.5)

    if feeders is not None and odour_spread is not None:
        if not isinstance(feeders, list):
            feeders = [feeders]
        for i, feeder in enumerate(feeders):
            xy = odour_spread[i + 1] * np.exp(1j * np.linspace(0, 2 * np.pi, 101, endpoint=True))
            ax.plot(xy.real + feeder[1], xy.imag + feeder[0], f'C{i+1}--', lw=2, alpha=.5)

    if world is not None:
        for polygon, colour in zip(world.polygons, world.colours):
            x = polygon[[0, 1, 2, 0], 0]
            y = polygon[[0, 1, 2, 0], 1]
            ax.fill_between(y, x[0], x, facecolor=colour, edgecolor=colour, alpha=.7, lw=.5)

    if nest is not None:
        ax.scatter([nest[1]], [nest[0]], marker='o', s=50, c='black')
        ax.text(nest[1] - .5, nest[0] + .2, "Nest")

    feeders_text = []
    if feeders is not None:
        if not isinstance(feeders, list):
            feeders = [feeders]
        for i, feeder in enumerate(feeders):
            ax.scatter([feeder[1]], [feeder[0]], marker='o', s=50, c='black')
            feeder_name = "Feeder" if len(feeders) < 2 else f"{chr(ord('A') + i)}"
            ax.text(feeder[1] + .2, feeder[0] + .2, feeder_name)
            text = ax.text(feeder[1] - .7, feeder[0] + .5, "", ha='left', va='bottom')
            feeders_text.append(text)

    ax.set_ylim(0, 10)
    ax.set_xlim(0, 10)
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', labelsize=8)

    cal = ax.scatter([], [], marker='.', s=50, c='orange')

    poi = ax.scatter([], [], marker='.', s=100, c='blue')

    line_c, = ax.plot([], [], 'r', lw=2)
    pos = ax.scatter([], [], marker=(3, 2, 0), s=100, c='red')

    points = [0, 2, 3, 4, 6]
    vert = np.array(pos.get_paths()[0].vertices)[points]
    vert[0] *= 2
    codes = pos.get_paths()[0].codes[points]
    vert = np.hstack([vert, np.zeros((vert.shape[0], 1))])

    return line_c, line_b, pos, (vert, codes), cal, poi, feeders_text


def create_side_axis(world=None, subplot=111, ax=None):
    """
    Draws a map with all the vegetation from the world (if given), the nest and feeder positions (if given) and returns
    the ongoing and previous paths of the agent, the agent's current position, the marker (arror) of the agents facing
    direction, the calibration points and the points where the agent is taken back on the route after replacing.

    Parameters
    ----------
    world: Seville2009, optional
        the world containing the vegetation. Default is None
    nest: np.ndarray[float], optional
        the position of the nest. Default is None
    feeder: np.ndarray[float], optional
        the position of the feeder. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    line_c: matplotlib.lines.Line2D
        the ongoing path of the agent
    line_b: matplotlib.lines.Line2D
        the previous paths of the agent
    pos: matplotlib.collections.PathCollection
        the current position of the agent
    marker: tuple[np.ndarray[float], np.ndarray[float]]
        the marker parameters
    cal: matplotlib.collections.PathCollection
        the points on the map where the calibration took place
    poi: matplotlib.collections.PathCollection
        the points on the map where the agent was put back on the route
    """

    if ax is None:
        ax = plt.subplot(subplot)

    ymin, ymax = None, None
    zmin, zmax = None, None
    if world is not None:
        for polygon, colour in zip(world.polygons, world.colours):
            z = polygon[[0, 1, 2, 0], 2]
            y = polygon[[0, 1, 2, 0], 1]
            z_min, z_max = np.min(z), np.max(z)
            y_min, y_max = np.min(y), np.max(y)
            if zmin is None or z_min < zmin:
                zmin = z_min
            if zmax is None or z_max > zmax:
                zmax = z_max
            if ymin is None or y_min < ymin:
                ymin = y_min
            if ymax is None or y_max > ymax:
                ymax = y_max
            ax.plot(y, z, c=colour)

    ax.set_aspect('equal', 'box')
    ax.set_ylim(min(0, zmin), zmax * 1.1)
    ax.set_xlim(ymin - .5, ymax + .5)
    ax.tick_params(axis='both', labelsize=8)


def create_eye_axis(eye, cmap="Greys_r", subplot=111, ax=None):
    """
    Draws a map of the positions of the ommatidia coloured using their photo-receptor responses.

    Parameters
    ----------
    eye: CompoundEye
        the eye to take the ommatidia positions from
    cmap: str, optional
        the colour map of the responses. Default is 'Greys_r'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.collections.PathCollection
        the ommatidia as a path collection
    """
    if ax is None:
        ax = plt.subplot(subplot)
    ax.set_yticks(np.sin([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]))
    ax.set_yticklabels([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-180, 180)
    ax.tick_params(axis='both', labelsize=8)

    yaw, pitch, roll = eye.omm_ori.as_euler('ZYX', degrees=True).T
    eye_size = 5000. / eye.nb_ommatidia * eye.omm_area * 80
    omm = ax.scatter(yaw.tolist(), (np.sin(np.deg2rad(-pitch))).tolist(), s=eye_size,
                     c=np.zeros(yaw.shape[0], dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    return omm


def create_sphere_eye_axis(eye, cmap="Greys_r", side="top", subplot=111, ax=None):
    """
    Draws a map of the positions of the ommatidia coloured using their photo-receptor responses.

    Parameters
    ----------
    eye: CompoundEye
        the eye to take the ommatidia positions from
    cmap: str, optional
        the colour map of the responses. Default is 'Greys_r'
    side: str, optional
        specifies whether to show the 'top' or 'side' of the eye. Default is 'top'.
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.collections.PathCollection
        the ommatidia as a path collection
    """
    if ax is None:
        ax = plt.subplot(subplot)

    x_, y_, z_ = eye.omm_xyz.T
    if side == "top":
        select = z_ >= 0
        x = -x_[select]
        y = -y_[select]
    else:
        select = y_ >= 0
        x = -x_[select]
        y = z_[select]
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='both', labelsize=8)

    eye_size = 5000. / x.size * eye.omm_area[select] * 80
    omm = ax.scatter(x.tolist(), y.tolist(), s=eye_size,
                     c=np.zeros(x.shape[0], dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    return omm


def create_mem_axis(agent, cmap="Greys", subplot=111, ax=None):
    """
    Draws the responses of the PNs, KCs and the familiarity current value in neuron-like arrays.

    Parameters
    ----------
    agent: VisualNavigationAgent
        The agent to get the data from
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    pn: matplotlib.collections.PathCollection
        collection of the PN responses
    kc: matplotlib.collections.PathCollection
        collection of the KC responses
    fam: matplotlib.collections.PathCollection
        collection of the familiarity value per scan
    """
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 13)
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    mem = agent.memory_component
    nb_pn = mem.nb_input
    nb_kc = mem.nb_hidden

    size = 400.
    ax.text(.1, 4.8, "PN", fontsize=10)
    pn = ax.scatter(np.linspace(.3, 12.7, nb_pn), np.full(nb_pn, 4.5), s=size / nb_pn,
                    c=np.zeros(nb_pn, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 3.8, "KC", fontsize=10)
    nb_rows = 50
    nb_cols = int(nb_kc / nb_rows) + 1
    x = np.array([np.linspace(.3, 12.7, nb_cols)] * nb_rows).flatten()[:nb_kc]
    y = np.array([np.linspace(1.3, 3.5, nb_rows)] * nb_cols).T.flatten()[:nb_kc]
    kc = ax.scatter(x, y, s=size / nb_kc, c=np.zeros(nb_kc, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 0.8, "familiarity", fontsize=10)
    nb_fam = len(agent.pref_angles)
    fam = ax.scatter(np.linspace(.3, 12.7, nb_fam), np.full(nb_fam, 0.5), s=size / nb_fam,
                     c=np.zeros(nb_fam, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    return pn, kc, fam


def create_pn_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the PN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: VisualNavigationAgent | NavigationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the PN history responses
    """
    if isinstance(agent, VisualNavigationAgent):
        nb_pn = agent.memory_component.nb_input
    else:
        nb_pn = agent.mushroom_body.nb_cs
    return create_image_history(nb_pn, nb_frames, sep=sep, title="PN",  cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_kc_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the KC history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: VisualNavigationAgent | NavigationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the KC history responses
    """
    if isinstance(agent, VisualNavigationAgent):
        nb_kc = agent.memory_component.nb_hidden
    else:
        nb_kc = agent.mushroom_body.nb_kc
    return create_image_history(nb_kc, nb_frames, sep=sep, title="KC",  cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_mbon_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the MBON history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: NavigationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the MBON history responses
    """
    nb_mbon = agent.mushroom_body.nb_mbon
    return create_image_history(nb_mbon, nb_frames, sep=sep, title="MBON", cmap=cmap, vmin=-2, vmax=2, subplot=subplot, ax=ax)


def create_dan_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the DAN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: NavigationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the DAN history responses
    """
    nb_dan = agent.mushroom_body.nb_dan
    return create_image_history(nb_dan, nb_frames, sep=sep, title="DAN", cmap=cmap, vmin=-2, vmax=2, subplot=subplot, ax=ax)


def create_familiarity_response_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the familiarity history for every scan as an image, where each pixel is a scan in time and its colour reflects
    the familiarity in this scan. Also the lowest value is marked using a red line.

    Parameters
    ----------
    agent: VisualNavigationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the familiarity history
    matplotlib.lines.Line2D
        the line showing the lowest familiarity value
    """
    nb_scans = agent.nb_scans
    if nb_scans <= 1:
        nb_scans = agent.nb_mental_rotations

    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, nb_scans-1)
    ax.set_xlim(0, nb_frames-1)
    ax.set_yticks([0, nb_scans//2, nb_scans-1])
    angles = np.roll(agent.pref_angles, len(agent.pref_angles) // 2)
    ax.set_yticklabels([angles[0], angles[len(angles) // 2], angles[-1]])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("familiarity", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    fam = ax.imshow(np.zeros((nb_scans, nb_frames), dtype='float32'), cmap=cmap,
                    vmin=0, vmax=1,
                    # vmin=0.40, vmax=0.60,
                    interpolation="none", aspect="auto")

    fam_line, = ax.plot([], [], 'red', lw=.5, alpha=.5)

    if sep is not None:
        ax.plot([sep, sep], [0, nb_scans-1], 'grey', lw=1)

    return fam, fam_line


def create_familiarity_map(nb_cols, nb_rows, cmap="RdPu", subplot=111, ax=None):
    """
    Draws the familiarity history for every scan as an image, where each pixel is a scan in time and its colour reflects
    the familiarity in this scan. Also the lowest value is marked using a red line.

    Parameters
    ----------
    agent: VisualNavigationAgent
        the agent to get the data and properties from
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the familiarity history
    matplotlib.lines.Line2D
        the line showing the lowest familiarity value
    """
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, nb_rows-1)
    ax.set_xlim(0, nb_cols-1)
    ax.set_axis_off()
    ax.set_aspect('auto', 'box')
    ax.set_ylabel("familiarity", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    fam = ax.imshow(np.zeros((nb_cols, nb_rows), dtype='float32'), cmap=cmap, vmin=0.0, vmax=1.0,
                    interpolation="none", aspect="auto")
    x, y = np.arange(nb_cols), np.arange(nb_rows)
    x, y = np.meshgrid(x, y)
    v, u = np.zeros_like(x), np.zeros_like(y)
    qui = ax.quiver(x, y, v, u, pivot='mid', color='k', scale=15)

    return fam, qui


def create_familiarity_history(nb_frames, sep=None, subplot=111, ax=None):
    """
    Draws a line of the lowest familiarity per iteration.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the line of the lowest familiarity per iteration
    """
    return create_single_line_history(nb_frames, sep=sep, title="familiarity (%)", ylim=100, subplot=subplot, ax=ax)


def create_free_space_history(nb_frames, sep=None, subplot=111, ax=None):
    """
    Draws a line of the available capacity per iteration.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the line of the available capacity per iteration.
    """
    return create_single_line_history(nb_frames, sep=sep, title="free space (%)", ylim=100, subplot=subplot, ax=ax)

def create_dra_axis(sensor, cmap="coolwarm", centre=None, scale=1., draw_axis=True, subplot=111, ax=None):
    """
    Draws the DRA and the responses of its ommatidia.

    Parameters
    ----------
    sensor: PolarisationSensor
        the compass sensor to get the data and parameters from
    centre: list[float], optional
        the centre of the DRA map. Default is [.5, .5]
    scale: float, optional
        a factor that scales the position of the ommatidia on the figure. Default is 1
    draw_axis: bool, optional
        if True, it draws the axis for the DRA, otherwise it draws on the existing axis. Default is True
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.collections.PathCollection
        the ommatidia of the DRA as a path collection
    """
    x, y, _ = sensor.omm_ori.apply(np.array([1, 0, 0])).T
    omm_y = -x
    omm_x = y

    if ax is None:
        ax = plt.subplot(subplot)

    if centre is None:
        centre = [.5, .5]

    if draw_axis:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    size = 20. * scale
    ax.text(centre[0] - .7, centre[1] + .3, "POL", fontsize=10)
    omm = ax.scatter((omm_y * scale + centre[0]).tolist(), (omm_x * scale + centre[1]).tolist(), s=size,
                     c=np.zeros(omm_y.shape[0], dtype='float32'), cmap=cmap, vmin=-.5, vmax=.5)

    return omm

def create_dra_axis_minimal_device(sensor, cmap="coolwarm", centre=None, scale=1., draw_axis=True, subplot=111, ax=None, flip=False):
    """
    Draws the DRA and the responses of its ommatidia.

    Parameters
    ----------
    sensor: PolarisationSensor
        the compass sensor to get the data and parameters from
    centre: list[float], optional
        the centre of the DRA map. Default is [.5, .5]
    scale: float, optional
        a factor that scales the position of the ommatidia on the figure. Default is 1
    draw_axis: bool, optional
        if True, it draws the axis for the DRA, otherwise it draws on the existing axis. Default is True
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None
    flip: bool, optional
          switch the x and y axes in the plot. Default is False
    Returns
    -------
    matplotlib.collections.PathCollection
        the ommatidia of the DRA as a path collection
    """
    x, y, _ = sensor.omm_ori.apply(np.array([1, 0, 0])).T
    omm_y = -x
    omm_x = y

    if ax is None:
        ax = plt.subplot(subplot)

    if centre is None:
        centre = [.5, .5]

    if draw_axis:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    size = 20. * scale
    ax.text(centre[0] - .3, centre[1] + .3, "POL neuron locations", fontsize=10)
    n_neurons = len(omm_y)
    n_neurons_per_semicircle = int(len(omm_y)/2)
    point_idx_labels = np.arange(1, n_neurons+1)
    inter_neuron_angle = 360 / n_neurons
    point_angle_labels = np.arange(
                        -inter_neuron_angle * n_neurons_per_semicircle,
                        inter_neuron_angle * (n_neurons_per_semicircle + 1),
                        inter_neuron_angle
    ).astype(int)
    if point_angle_labels[-1] == 180:
        point_angle_labels = point_angle_labels[:-1]

    ax.scatter(centre[0], centre[1], c='black')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    if flip:
        omm = ax.scatter(
            (omm_x * scale + centre[0]).tolist(),
            (omm_y * scale + centre[1]).tolist(), s=size,
            vmin=-.5, vmax=.5, c=colors[:n_neurons]
        )
        for i, txt in enumerate(point_idx_labels):
            ax.annotate(f"P{txt}", ((omm_x * scale + centre[0]).tolist()[i], (omm_y * scale + centre[0]).tolist()[i]), color=colors[i])
        for i, txt in enumerate(point_angle_labels):
            ax.annotate(" "*4+f"({txt}\xb0)", ((omm_x * scale + centre[0]).tolist()[i], (omm_y * scale + centre[0]).tolist()[i]), color=colors[i])
    else:
        omm = ax.scatter(
                (omm_y * scale + centre[0]).tolist(),
                (omm_x * scale + centre[1]).tolist(), s=size,
                vmin=-.5, vmax=.5, c=colors[:n_neurons]
        )
        for i, txt in enumerate(point_idx_labels):
            ax.annotate(f"P{txt}", ((omm_y * scale + centre[0] - 0.05).tolist()[i], (omm_x * scale + centre[0] + 0.05).tolist()[i]), color=colors[i])
        for i, txt in enumerate(point_angle_labels):
            ax.annotate(" "*4+f"({txt}\xb0)", ((omm_y * scale + centre[0] - 0.05).tolist()[i], (omm_x * scale + centre[0] + 0.05).tolist()[i]), color=colors[i])
    return omm


def create_cmp_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the compass history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the TB1 history responses
    """
    nb_cmp = agent.brain[2].nb_cmp
    return create_image_history(nb_cmp, nb_frames, sep=sep, title="CMP", cmap=cmap, vmin=0, vmax=1, subplot=subplot,
                                ax=ax)

def create_direction_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    nb_pol = agent._pol_sensor.nb_lenses
    return create_image_history(nb_pol, nb_frames, sep=sep, title="POL", cmap=cmap, vmin=0, vmax=1.5, subplot=subplot, ax=ax)

def create_memory_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    nb_memory = agent._cx.nb_memory
    return create_image_history(nb_memory, nb_frames, sep=sep, title="memory", cmap=cmap, vmin=-10.0, vmax=10.0, subplot=subplot, ax=ax)

def create_sigmoid_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    nb_sigmoid = agent._cx.nb_sigmoid
    return create_image_history(nb_sigmoid, nb_frames, sep=sep, title="sigmoid neuron", cmap=cmap, vmin=0.9, vmax=1.1, subplot=subplot, ax=ax)

def create_steering_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    nb_steering = agent._cx.nb_steering
    return create_image_history(nb_steering, nb_frames, sep=sep, title="steering", cmap=cmap, vmin=26490000, vmax=26560000, subplot=subplot, ax=ax)

def create_steering_diff_history(nb_frames, sep=None, subplot=111, ax=None):
    return create_single_line_history(nb_frames, sep=sep, title="right-left steering", ylim_lower=-2.5, ylim_upper=2.5, subplot=subplot, ax=ax)

def create_tb1_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the TB1 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the TB1 history responses
    """
    nb_tb1 = agent.central_complex.nb_tb1
    return create_image_history(nb_tb1, nb_frames, sep=sep, title="TB1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cl1_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CL1 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CL1 history responses
    """
    nb_cl1 = agent.central_complex.nb_cl1
    return create_image_history(nb_cl1, nb_frames, sep=sep, title="CL1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu1_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CPU1 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU1 history responses
    """
    nb_cpu1 = agent.central_complex.nb_cpu1
    return create_image_history(nb_cpu1, nb_frames, sep=sep, title="CPU1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CPU4 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU4 history responses
    """
    nb_cpu4 = agent.central_complex.nb_cpu4
    # return create_image_history(nb_cpu4, nb_frames, sep=sep, title="hΔC", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_vec_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the vector memory activation history of responses as an image, where each pixel is a neuron in time and its
    colour reflects the response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU4 history responses
    """
    nb_vec = agent.central_complex.nb_vectors
    return create_image_history(nb_vec, nb_frames, sep=sep, title="vectors", cmap=cmap, vmin=-1, vmax=1,
                                subplot=subplot, ax=ax)


def create_cpu4_mem_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CPU4 history of memories as an image, where each pixel is a neuron in time and its colour reflects the
    memory of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU4 history memories
    """
    nb_cpu4 = agent.central_complex.nb_cpu4
    # return create_image_history(nb_cpu4, nb_frames, sep=sep, title="PFN", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4 (mem)", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_compass_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the compass history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the compass history responses
    """
    nb_cmp = agent.brain[1].nb_cmp
    return create_image_history(nb_cmp, nb_frames, sep=sep, title="Compass", cmap=cmap, vmin=-1, vmax=1,
                                subplot=subplot, ax=ax)


def create_epg_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the E-PG history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the E-PG history responses
    """
    nb_epg = agent.brain[2].nb_epg
    return create_image_history(nb_epg, nb_frames, sep=sep, title="E-PG", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_peg_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the P-EG history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the P-EG history responses
    """
    nb_peg = agent.brain[2].nb_peg
    return create_image_history(nb_peg, nb_frames, sep=sep, title="P-EG", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_pen_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the P-EN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the P-EN history responses
    """
    nb_pen = agent.brain[2].nb_pen
    return create_image_history(nb_pen, nb_frames, sep=sep, title="P-EN", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_pfl_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the PFL3 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep:  float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the PFL3 history responses
    """
    nb_pfl = agent.brain[2].nb_pfl3
    return create_image_history(nb_pfl, nb_frames, sep=sep, title="PFL3", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_fbn_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the FsBN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep:  float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the FsBN history responses
    """
    nb_fbn = agent.brain[2].nb_fbn
    return create_image_history(nb_fbn, nb_frames, sep=sep, title="FsBN", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_bcx_axis(agent, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws all the neurons and ommatidia of the given agent in a single axis, representing a snapshot of their current
    values.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    omm: matplotlib.collections.PathCollection
        the ommatidia of the DRA as a path collection
    tb1: matplotlib.collections.PathCollection
        the TB1 responses as a path collection
    cl1: matplotlib.collections.PathCollection
        the CL1 responses as a path collection
    cpu1: matplotlib.collections.PathCollection
        the CPU1 responses as a path collection
    cpu4: matplotlib.collections.PathCollection
        the CPU4 responses as a path collection
    cpu4mem: matplotlib.collections.PathCollection
        the CPU4 memories as a path collection
    """
    if ax is None:
        ax = plt.subplot(subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 5)
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    size = 20.
    omm = create_dra_axis(agent.pol_sensor, cmap=cmap, centre=[.8, 4.5], draw_axis=False, ax=ax)

    ax.text(1.5, 4.8, "TB1", fontsize=10)
    tb1 = ax.scatter(np.linspace(2, 4.5, 8), np.full(8, 4.5), s=2 * size,
                     c=np.zeros_like(agent.central_complex.r_tb1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 3.8, "CL1", fontsize=10)
    cl1 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 3.5), s=2 * size,
                     c=np.zeros_like(agent.central_complex.r_cl1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 2.8, "CPU1", fontsize=10)
    cpu1 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 2.5), s=2 * size,
                      c=np.zeros_like(agent.central_complex.r_cpu1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 1.8, "CPU4", fontsize=10)
    cpu4 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 1.5), s=2 * size,
                      c=np.zeros_like(agent.central_complex), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, .8, "CPU4 (mem)", fontsize=10)
    cpu4mem = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, .5), s=2 * size,
                         c=np.zeros_like(agent.central_complex.r_cpu4), cmap=cmap, vmin=0, vmax=1)

    return omm, tb1, cl1, cpu1, cpu4, cpu4mem


def create_image_history(nb_values, nb_frames, sep=None, title=None, cmap="Greys", subplot=111, vmin=0, vmax=1,
                         ax=None):
    """
    Draws the history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    nb_values: int
    nb_frames: int
        the total number of frames for the animation
    title: str, optional
    vmin: float, optional
    vmax: float, optional
    sep: float, np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the history responses
    """
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-0.5, nb_values-0.5)
    ax.set_xlim(-0.5, nb_frames-0.5)
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if title is not None:
        ax.set_ylabel(title)

    im = ax.imshow(np.zeros((nb_values, nb_frames), dtype='float32'), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="none", aspect="auto")

    if sep is not None:
        if isinstance(sep, numbers.Number):
            sep = [sep]
        for s in sep:
            ax.plot([s, s], [0, nb_values-1], 'grey', lw=1)

    return im


def create_single_line_history(nb_frames, sep=None, title=None, ylim_lower=0., ylim_upper=1., subplot=111, ax=None):
    """
    Draws a single line representing the history of a value.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    title: str, optional
        draw the title for the plot. Default is None
    ylim: float, optional
        the maximum value for the Y axis. Default is 1
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the drawn line
    """
    return create_multi_line_history(nb_frames, 1, sep=sep, title=title, ylim_lower=ylim_lower, ylim_upper=ylim_upper, subplot=subplot, ax=ax)


def create_multi_line_history(nb_frames, nb_lines, sep=None, title=None, ylim_lower=0, ylim_upper=1., subplot=111, ax=None):
    """
    Draws multiple lines representing the history of many values.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    nb_lines: int
    title: str, optional
        draw the title for the plot. Default is None
    ylim: float, optional
        the maximum value for the Y axis. Default is 1
    sep: float | np.ndarray[float]
        the iteration(s) where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the drawn lines
    """
    ax = get_axis(ax, subplot)

    ax.set_ylim(ylim_lower, ylim_upper)
    ax.set_xlim(0, nb_frames)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_aspect('auto', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axhline(int((ylim_lower+ylim_upper)/2),c='red')

    if sep is not None:
        if isinstance(sep, numbers.Number):
            sep = [sep]
        for s in sep:
            ax.plot([s, s], [ylim_lower, ylim_upper], 'grey', lw=3)

    lines, = ax.plot(np.full((nb_frames, nb_lines), np.nan), 'k-', lw=2)
    if title is not None:
        ax.text(120, ylim_upper * 1.05, title, fontsize=10)

    return lines


def get_axis(ax=None, subplot=111):
    """
    If the axis is None it creates a new axis in the 'subplot' slot, otherwise it returns the given axis.

    Parameters
    ----------
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        if isinstance(subplot, int):
            ax = plt.subplot(subplot)
        else:
            ax = plt.subplot(*subplot)
    return ax


def col2x(col, nb_cols, max_meters=10.):
    """
    Transforms column number to the 'x' coordinate.

    Parameters
    ----------
    col : int
        the column number
    nb_cols : int
        the total number of columns
    max_meters : float
        the maximum width of the arena in meters. Default is 10 meters

    Returns
    -------
    float
        the 'x' coordinate
    """
    return np.float32(col * max_meters) / np.float32(nb_cols)


def x2col(x, nb_cols, max_meters=10.):
    """
    Transforms the 'x' coordinate to the column number.

    Parameters
    ----------
    x : float
        the 'x' coordinate
    nb_cols : int
        the total number of columns
    max_meters : float
        the maximum width of the arena in meters. Default is 10 meters

    Returns
    -------
    int :
        the column number
    """
    return np.int(np.float32(x * nb_cols) / np.float32(max_meters))


def row2y(row, nb_rows, max_meters=10.):
    """
    Transforms row number to the 'y' coordinate.

    Parameters
    ----------
    row : int
        the row number
    nb_rows : int
        the total number of rows
    max_meters : float
        the maximum length of the arena in meters. Default is 10 meters

    Returns
    -------
    float
        the 'y' coordinate
    """
    return np.float32(row * max_meters) / np.float32(nb_rows)


def y2row(y, nb_rows, max_meters=10):
    """
    Transforms the 'y' coordinate to the row number.

    Parameters
    ----------
    y : float
        the 'y' coordinate
    nb_rows : int
        the total number of rows
    max_meters : float
        the maximum length of the arena in meters. Default is 10 meters

    Returns
    -------
    int :
        the row number
    """
    return np.int(np.float32(y * nb_rows) / np.float32(max_meters))


def ori2yaw(ori, nb_oris, linear=True, degrees=True):
    """
    Transforms the orientation identity to the yaw direction.

    Parameters
    ----------
    ori : int
        the orientation identity
    nb_oris : int
        the total number of orientations
    linear : bool
        whether the ori2yaw map is linear or not
    degrees : bool
        whether we want the output to be in degrees

    Returns
    -------
    float
        the yaw direction
    """

    two_pi = np.float32(360. if degrees else (2 * np.pi))
    if not linear:
        ori = ((ori / nb_oris - .25) ** 3 + .25) * nb_oris
    return (np.float32(ori) * two_pi / np.float32(nb_oris) + two_pi / 2) % two_pi - two_pi / 2


def yaw2ori(yaw, nb_oris, linear=True, degrees=True):
    """
    Transforms the 'yaw' direction to the orientation identity.

    Parameters
    ----------
    yaw : float
        the 'yaw' direction
    nb_oris : int
        the total number of orientations
    linear : bool
        whether the ori2yaw map is linear or not
    degrees : bool
        whether the input is in degrees

    Returns
    -------
    int :
        the orientation identity
    """
    two_pi = np.float32(360. if degrees else (2 * np.pi))
    ori = np.float32((yaw % two_pi) * nb_oris) / two_pi
    if not linear:
        if ori / nb_oris >= .25:
            ori = (np.power(ori / nb_oris - .25, 1/3) + .25) * nb_oris
        else:
            ori = (-np.power(np.abs(ori / nb_oris - .25), 1/3) + .25) * nb_oris
    return np.int(ori)
