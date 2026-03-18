from invertsy.env.world import Seville2009
from invertsy.agent.agent import MinimalDeviceCentralComplexAgent
from invertsy.sim.minimal_device_simulation import MinimalDevicePathIntegrationSimulation
from invertsy.sim.animation import PathIntegrationAnimation
from invertsy.sim.minimal_device_animation import MinimalDevicePathIntegrationAnimation
import numpy as np

def main(*args):
    routes = Seville2009.load_routes(args[0], degrees=True)

    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    rt = rt[::-1][:400]
    rt[:, 3] = (rt[:, 3] - 0) % 360 - 180

    # Create parameter dictionary
    spiking = False
    use_nanowires = True
    use_dye = True
    sigmoid_bool = True
    nanowire_sigmoid = True
    nanowire_sigmoid_devs = np.arange(0,0.15,0.05)
    communication_downscaling_factors = np.arange(1, 11, 1)  # value by which to divide weights
    communication_noise_factors = np.arange(0, 2.5, 0.5)  # percentage of max value
    nanowire_distances = np.arange(0,8,1)  # micrometres
    unit_distance = 7  # micrometres
    transmittances_per_distance_oom = nanowire_distances / unit_distance  # order of magnitude of transmittance decrease
    transmittance_downscaling_factors = 10**transmittances_per_distance_oom

    total_downscaling_factors = []
    for communication_downscaling_factor in communication_downscaling_factors:
        for transmittance_downscaling_factor in transmittance_downscaling_factors:
            total_downscaling_factor = transmittance_downscaling_factor * communication_downscaling_factor
            total_downscaling_factors.append(total_downscaling_factor)
    total_downscaling_factors = np.unique(total_downscaling_factors)
    total_downscaling_factors = np.arange(1,21,2)

    for communication_noise_factor in communication_noise_factors:
        for total_downscaling_factor in total_downscaling_factors:
            for nanowire_sigmoid_dev in nanowire_sigmoid_devs:
                cx_params = {"spiking":spiking,
                             "use_nanowires": use_nanowires,
                         "sigmoid_bool": sigmoid_bool,
                         "nanowire_sigmoid": nanowire_sigmoid,
                         "nanowire_sigmoid_dev": nanowire_sigmoid_dev,
                         "use_dye": use_dye,
                         #"communication_downscaling_factor": communication_downscaling_factor,
                         "communication_noise_factor": communication_noise_factor,
                         #"transmittance_per_distance_oom": transmittance_downscaling_factor,
                         "total_downscaling_factor": total_downscaling_factor}
                print('cx params', cx_params)

                agent = MinimalDeviceCentralComplexAgent(cx_params=cx_params)
                agent.step_size = .01
                sim = MinimalDevicePathIntegrationSimulation(rt, nanowire_sigmoid_dev, communication_noise_factor, total_downscaling_factor, agent=agent, noise=0., name="pi-ant%d-route%d" % (ant_no, rt_no))
                ani = MinimalDevicePathIntegrationAnimation(sim, show_history=True)
                ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import argparse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        parser = argparse.ArgumentParser(
            description="Run a path integration test."
        )

        parser.add_argument("-i", dest='input', type=str, required=False, default=Seville2009.ROUTES_FILENAME,
                            help="File with the recorded routes.")

        p_args = parser.parse_args()
        main(p_args.input)
