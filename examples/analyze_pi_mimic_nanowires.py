from invertsy.env.world import Seville2009
from invertsy.agent.agent import MinimalDeviceCentralComplexAgent
from invertsy.sim.minimal_device_simulation import MinimalDevicePathIntegrationSimulation
from invertsy.sim.animation import PathIntegrationAnimation
from invertsy.sim.minimal_device_animation import MinimalDevicePathIntegrationAnimation
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    nanowire_sigmoid = True
    communication_downscaling_factors = np.arange(1, 11, 1)  # percentage of value
    communication_noise_factors = np.arange(0, 2.5, 0.5)  # percentage of max value
    nanowire_distances = np.arange(0, 8, 1)  # micrometres
    unit_distance = 7  # micrometres
    transmittances_per_distance_oom = nanowire_distances / unit_distance  # order of magnitude of transmittance decrease
    transmittance_downscaling_factors = 10 ** transmittances_per_distance_oom
    nanowire_sigmoid_dev = 0.0

    total_downscaling_factors = []
    for communication_downscaling_factor in communication_downscaling_factors:
        for transmittance_downscaling_factor in transmittance_downscaling_factors:
            total_downscaling_factor = transmittance_downscaling_factor * communication_downscaling_factor
            total_downscaling_factors.append(total_downscaling_factor)
    total_downscaling_factors = np.unique(total_downscaling_factors)

    min_dists = []
    total_downscaling_factors = np.arange(1,21,2)
    for communication_noise_factor in communication_noise_factors:
        min_dists_per_noise = []
        for total_downscaling_factor in total_downscaling_factors:
            cx_params = {"use_nanowires": use_nanowires,
                         "sigmoid_bool": sigmoid_bool,
                         "nanowire_sigmoid": nanowire_sigmoid,
                         "use_dye": use_dye,
                         #"communication_downscaling_factor": communication_downscaling_factor,
                         "communication_noise_factor": communication_noise_factor,
                         #"transmittance_per_distance_oom": transmittance_downscaling_factor,
                         "total_downscaling_factor": total_downscaling_factor}
            print('cx params', cx_params)

            path = "C:\\Users\\rchit\\Documents\\Groningen\\Research\\InsectNeuroNano_InvertSy\\data\\results_minimal_device\\mimic_nanowires\\noise{}_totaldownscale{}_sigmoiddev{}.npy".format(communication_noise_factor,round(total_downscaling_factor,2),round(nanowire_sigmoid_dev,2))
            agent_locations = np.load(path)
            agent_locations = agent_locations[:len(agent_locations) // 3 * 3].reshape(len(agent_locations) // 3, 3)
            start = agent_locations[0,:2]
            end = agent_locations[813,:2]
            agent_locations = agent_locations[814:,:2]
            route_distance = np.linalg.norm(start-end)
            dists = np.linalg.norm(agent_locations - start, axis=1)
            min_dist = np.min(dists)
            min_dists_per_noise.append(100 * min_dist / route_distance)
        min_dists.append(min_dists_per_noise)
    print(min_dists)
    x_min,x_max = communication_noise_factors[0],communication_noise_factors[-1]
    y_min,y_max = total_downscaling_factors[:48][0],total_downscaling_factors[:48][-1]
    plt.imshow(min_dists[::-1],extent=[y_min, y_max, x_min, x_max])
    plt.xlabel('Total downscaling factor')
    plt.ylabel('Noise magnitude (%)')
    plt.title('Closest dist to nest (% of outbound route length)')
    cbar = plt.colorbar(shrink=0.4)  # shrink makes it less tall
    cbar.set_label("% of outbound length", rotation=270, labelpad=11)  # text next to bar

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)  # size = thickness, pad = spacing
    # cbar = plt.colorbar(im, cax=cax)
    #
    # # Shrink the colorbar (make it less tall)
    # cbar.ax.set_aspect(10)  # increase value to shrink vertically
    #
    # # Add a label (next to colorbar)
    # cbar.set_label("Intensity", rotation=270, labelpad=15)

    #plt.savefig('C:\\Users\\rchit\\Documents\\Groningen\\Research\\InsectNeuroNano_InvertSy\\data\\results_minimal_device\\mimic_nanowires\\graphs\\results_sigmoiddev{}.png'.format(nanowire_sigmoid_dev))
    #plt.scatter(agent_locations[:,1], agent_locations[:,0])

if __name__ == '__main__':
    import warnings
    import argparse
    print(8)
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
