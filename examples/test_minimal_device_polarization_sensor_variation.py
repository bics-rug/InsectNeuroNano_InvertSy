from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertsy.env.sky import Sky
from invertsy.sim._helpers import create_dra_axis

from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

fov = 56
nb_ommatidia = 3
omm_photoreceptor_angle = 1
sensor = MinimalDevicePolarisationSensor(
            field_of_view=fov, nb_lenses=nb_ommatidia,
            omm_photoreceptor_angle=omm_photoreceptor_angle, omm_rho=np.deg2rad(5),
            omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
            omm_pol_op=1., noise=0.
        )

print(sensor)
print(sensor.omm_xyz)

sun_elevation = np.deg2rad(30)
all_r = []
sun_azimuths = np.arange(-np.pi, np.pi, np.pi/100)
for sun_azimuth in sun_azimuths:
    print(sun_azimuth)
    sky = Sky(sun_elevation, sun_azimuth)
    r = sensor(sky=sky)
    all_r.append(r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

all_r = np.array(all_r)
all_r = all_r.reshape((all_r.shape[0], all_r.shape[1]))
n_ommatidia = all_r.shape[1]
labels = [str(i) for i in range(1, n_ommatidia+1)]
for idx in range(n_ommatidia):
    ax1.plot(sun_azimuths, all_r[:, idx], label=labels[idx])
ax1.set_xlabel('Solar azimuth (rad)')
ax1.set_ylabel('POL neuron response')
ax1.legend(loc='upper left')

pol = create_dra_axis(sensor, draw_axis=True, ax=ax2, flip=False)
pol.set_array(np.array(r).flatten())

plt.subplots_adjust(wspace=0.4)

plt.show()
