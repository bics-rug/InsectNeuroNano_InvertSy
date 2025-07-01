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

    rt = rt[::-1]
    rt[:, 3] = (rt[:, 3] - 0) % 360 - 180

    # Create parameter dictionary
    use_nanowires = True
    use_dye = True
    sigmoid_bool = True
    communication_downscaling_factors = np.arange(50, 105, 5)  # percentage of value
    communication_noise_factors = np.arange(0, 55, 5)  # percentage of max value
    nanowire_distance = 10  # micrometres
    unit_distance = 7  # micrometres
    transmittance_per_distance_oom = nanowire_distance / unit_distance  # order of magnitude of transmittance decrease
    for i in range(len(communication_downscaling_factors)):
        for j in range(len(communication_noise_factors)):
            cx_params = {"use_nanowires": use_nanowires,
                         "sigmoid_bool": sigmoid_bool,
                         "use_dye": use_dye,
                         "communication_downscaling_factor": communication_downscaling_factors[i],
                         "communication_noise_factor": communication_noise_factors[j],
                         "transmittance_per_distance_oom": transmittance_per_distance_oom}
            print('cx params', cx_params)

            agent = MinimalDeviceCentralComplexAgent(cx_params=cx_params)
            agent.step_size = .01
            sim = MinimalDevicePathIntegrationSimulation(rt, communication_downscaling_factors[i], communication_noise_factors[j], transmittance_per_distance_oom, agent=agent, noise=0., name="pi-ant%d-route%d" % (ant_no, rt_no))
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

        # # Create parameter dictionary
        # use_nanowires = True
        # use_dye = True
        # sigmoid_bool = True
        # communication_downscaling_factors = np.arange(20,105,5)  # percentage of value
        # communication_noise_factors = np.arange(0,50,5)  # percentage of max value
        # nanowire_distance = 10 # micrometres
        # unit_distance = 7 # micrometres
        # transmittance_per_distance_oom = nanowire_distance / unit_distance # order of magnitude of transmittance decrease
        # for df in communication_downscaling_factors:
        #     for nf in communication_noise_factors:
        #         cx_params = {"use_nanowires": use_nanowires,
        #              "sigmoid_bool": sigmoid_bool,
        #              "use_dye": use_dye,
        #              "communication_downscaling_factor": df,
        #              "communication_noise_factor": nf,
        #              "transmittance_per_distance_oom":transmittance_per_distance_oom}
        #         print('cx params',cx_params)
        main(p_args.input)
