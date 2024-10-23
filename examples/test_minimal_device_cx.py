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
    sun_elevation = np.deg2rad(45)
    sun_azimuth = np.deg2rad(0)
    sky = Sky(sun_elevation, sun_azimuth)

    # create polarization sensor
    POL_method = "single_0"
    fov = 120
    nb_ommatidia = 6
    omm_photoreceptor_angle = 2
    sensor = MinimalDevicePolarisationSensor(
        POL_method=POL_method,
        field_of_view=fov, nb_lenses=nb_ommatidia,
        omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
        omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
        omm_pol_op=1., noise=0.
    )

    # create central complex compass + integrator
    cx = MinimalDeviceCX(update=True)
    print(cx)

    # # rotate the compass in a full circle anti-clockwise,
    # # then in a full circle clockwise
    # yaws, steering_responses = [], []
    # for i in np.linspace(-360, 360, 73):
    #     POL_direction = sensor(sky=sky)
    #     steering = cx(POL_direction)
    #     steering_responses.append(steering)
    #     yaws.append(sensor.yaw_deg % 360)
    #     sensor.rotate(R.from_euler('ZYX', [np.sign(i) * 10, 0, 0], degrees=True))
    # steering_responses = np.array(steering_responses)
    #
    # # print the difference between the left and right steering responses
    # # as the compass is being rotated
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    # ax1.plot(np.linspace(0, -360, 37), steering_responses[:37,0] - steering_responses[:37,1])
    # ax1.set_xlim(20, -380)
    # ax1.set_title('ACW compass rotation -> steering right')
    # ax2.plot(np.linspace(0, 360, 37), steering_responses[36:,0] - steering_responses[36:,1])
    # ax2.set_title('CW compass rotation -> steering left')
    #
    # ax1.set_ylabel('Left-right steering response',fontsize=12)
    # plt.suptitle('Left-right steering response vs. compass turning angle')
    pred_angles=[]
    mem_angles=[]
    yaws, steering_responses = [], []
    for i in np.linspace(0, 360, 37)[:-1]:
        print(i)
        for _ in range(10):
            sky = Sky(sun_elevation, sun_azimuth)
            POL_direction = sensor(sky=sky)
            steering = cx(POL_direction)
            steering_responses.append(steering)
            yaws.append(sensor.yaw_deg % 360)
            vectors1,vectors2=[],[]
            thetas_6 = [-180, -120, -60, 0, 60, 120]
            thetas_3 = [-120, 0, 120]
            thetas_6 = [np.deg2rad(el) for el in thetas_6]
            thetas_3 = [np.deg2rad(el) for el in thetas_3]

            for idx in range(3):
                vectors1.append(POL_direction[idx] * np.exp(-1j * thetas_3[idx]))
                vectors2.append(-cx.memory.r_memory[idx] * np.exp(-1j * thetas_3[idx]))
            summed_vector1 = np.array(vectors1).sum()
            predicted_angle1 = np.angle(summed_vector1)
            summed_vector2 = np.array(vectors2).sum()
            predicted_angle2 = np.angle(summed_vector2)
            pred_angles.append(np.rad2deg(predicted_angle1))
            mem_angles.append(np.rad2deg(predicted_angle2))
            #print(f"{sensor.yaw_deg % 360=},{pred_angles[-1]=}, {mem_angles[-1]=}, {steering_responses[-1]=}, {steering[ 0] - steering[ 1]=}")
            #print(sensor.yaw_deg % 360, POL_direction[1]/POL_direction[0], cx.memory.r_memory[1]/cx.memory.r_memory[0],cx.memory.r_memory, steering[0]-steering[1])




        sensor.rotate(R.from_euler('ZYX', [10, 0, 0], degrees=True))
    """
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

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
    ax1.plot(steering_responses[:740,0] - steering_responses[:740,1])
    ax1.axhline(0, color='blue', linestyle='--', linewidth=1)
    ax1.set_ylabel('Left-right steering response',fontsize=12,c='blue')
    ax1.set_xlabel('Step')
    xtick_positions = [90,270,450,630]
    xtick_labels = ['0->+90','+90->-90','-90->+90','+90->-90']
    #plt.xticks(xtick_positions, xtick_labels)

    yaws = np.array(yaws)
    yaws[yaws==360] = 0
    ax2 = ax1.twinx()
    ax2.plot(yaws[:740],c='orange')
    ax2.set_ylabel('Sensor yaw',c='orange',fontsize=12)

    plt.suptitle('Left-right steering response vs. compass turning angle')
    plt.show()
    """
    pred_angles=np.array(pred_angles)
    pred_angles = (360 + np.round(pred_angles)) % 360

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
    ax1.scatter(range(len(pred_angles)),pred_angles,c='blue',s=5)
    ax1.plot(range(len(mem_angles)),mem_angles,c='orange')

    steering_responses = np.array(steering_responses)
    ax2 = ax1.twinx()
    ax2.plot(range(len(pred_angles)),steering_responses[:,0] - steering_responses[:,1],c='green')
    ax2.axhline(0, color='green', linestyle='--', linewidth=1)

    xticks=[90,180,270,359]
    labels=[]
    for idx in xticks:
        ax1.scatter(idx,pred_angles[idx],s=15,c='red')
        ax1.scatter(idx,mem_angles[idx],s=15,c='red')
        ax2.scatter(idx,steering_responses[idx,0] - steering_responses[idx,1],s=15,c='red')
        plt.axvline(idx,c='red')
        labels.append(f"yaw={int(round(pred_angles[idx]))}, mem={int(round(mem_angles[idx]))}")

    plt.xticks(xticks,labels)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Angle (degrees)',c='blue',fontsize=10)
    ax2.set_ylabel('POL response units',c='green',fontsize=10)
    plt.title('Left-right steering response vs. sensor direction and memory angles')

    save_folder = f"..\\data\\results_minimal_device\\"
    #plt.savefig(save_folder+"steering_response.png")
    plt.show()

if __name__ == '__main__':
    main(*sys.argv)
