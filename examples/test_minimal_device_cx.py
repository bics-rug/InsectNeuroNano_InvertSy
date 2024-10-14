# # ! /usr/bin/env python
# -*- coding: utf-8 -*-

from invertpy.brain.centralcomplex.minimal_device import MinimalDeviceCX
from invertsy.env.sky import Sky
from invertpy.sense.polarisation import MinimalDevicePolarisationSensor

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def main(*args):
    # create sky instance
    sun_elevation = np.deg2rad(30)
    sun_azimuth = 0
    sky = Sky(sun_elevation, sun_azimuth)

    # create polarization sensor
    POL_method = "single_0"
    fov = 56
    nb_ommatidia = 3
    omm_photoreceptor_angle = 2
    sensor = MinimalDevicePolarisationSensor(
        POL_method=POL_method,
        field_of_view=fov, nb_lenses=nb_ommatidia,
        omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
        omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
        omm_pol_op=1., noise=0.
    )

    # create central complex compass + integrator
    cx = MinimalDeviceCX()
    print(cx)

    # rotate the compass in a full circle anti-clockwise,
    # then in a full circle clockwise
    yaws, steering_responses = [], []
    for i in np.linspace(-360, 360, 73):
        POL_direction = sensor(sky=sky)
        steering = cx(POL_direction)
        steering_responses.append(steering)
        yaws.append(sensor.yaw_deg % 360)
        sensor.rotate(R.from_euler('ZYX', [np.sign(i) * 10, 0, 0], degrees=True))
    steering_responses = np.array(steering_responses)

    # print the difference between the left and right steering responses
    # as the compass is being rotated
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    ax1.plot(np.linspace(0, -360, 37), steering_responses[:37,0] - steering_responses[:37,1])
    ax1.set_xlim(20, -380)
    ax1.set_title('ACW compass rotation -> steering right')
    ax2.plot(np.linspace(0, 360, 37), steering_responses[36:,0] - steering_responses[36:,1])
    ax2.set_title('CW compass rotation -> steering left')

    ax1.set_ylabel('Left-right steering response',fontsize=12)
    plt.suptitle('Left-right steering response vs. compass turning angle')

    yaws, steering_responses = [], []
    for i in np.linspace(0, 90, 10)[:-1]:
        for _ in range(10):
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
        sensor.rotate(R.from_euler('ZYX', [10, 0, 0], degrees=True))

    for i in np.linspace(90, -90, 19)[:-1]:
        for _ in range(10):
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
        sensor.rotate(R.from_euler('ZYX', [- 10, 0, 0], degrees=True))

    for i in np.linspace(-90, 90, 19)[:-1]:
        for _ in range(10):
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
        sensor.rotate(R.from_euler('ZYX', [ 10, 0, 0], degrees=True))

    for i in np.linspace(90, -90, 19)[:-1]:
        for _ in range(10):
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
        sensor.rotate(R.from_euler('ZYX', [ - 10, 0, 0], degrees=True))

    steering_responses = np.array(steering_responses)
    print(steering_responses.shape)
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
    ax1.plot(steering_responses[:,0] - steering_responses[:,1])
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax1.set_ylabel('Left-right steering response',fontsize=12)
    ax1.set_xlabel('Step')
    xtick_positions = [90,270,450,630]
    xtick_labels = ['0->+90','+90->-90','-90->+90','+90->-90']
    plt.xticks(xtick_positions, xtick_labels)

    plt.suptitle('Left-right steering response vs. compass turning angle')

    save_folder = f"..\\data\\results_minimal_device\\steering_response.png"
    # plt.savefig(save_folder)
    plt.show()

if __name__ == '__main__':
    main(*sys.argv)
