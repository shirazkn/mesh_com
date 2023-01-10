import getopt
import os
import sys
import time
import utm
import pylab

from ftl_player import NodeNetwork

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.pyplot import pause

plt.rcParams["figure.autolayout"] = True

TIMEKEEPER_MAC = '00:30:1a:4f:5b:2f'  # NodeNetworkEvaluator.timestamps records this drone's timestamps
EARTH_RADIUS = 6371000  # in meters


class NodeNetworkEvaluator(NodeNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Synchronized CSV data stored as lists
        self.node_data = {}
        self.node_macs = []
        self.timestamps = []

    def collect_data(self, folder_path, file_list):
        if file_list:
            # Reuse functions from ftl_player.py
            self._NodeNetwork__read_data(folder_path, file_list)
            self._NodeNetwork__sync_node_timestamps()
            self.node_macs = [node.get_mac() for node in self.network_nodes]

            # Get length of usable data
            total_time = 100000
            for node in self.network_nodes:
                if len(node._Node__f_rssi) < total_time:
                    total_time = len(node._Node__f_rssi)

            self.node_data = {node.get_mac(): {"neighbours": [], "rssi": [], "location": []}
                              for node in self.network_nodes}
            timestamps = []
            for _ in range(total_time):
                for node in self.network_nodes:
                    node.update_row_offset_from_seconds_offset(self.sec_offset_from_start)
                    self.node_data[node.get_mac()]["neighbours"].append([_str.split(',')[0] for _str in node.get_rssi().split(';')])
                    self.node_data[node.get_mac()]["rssi"].append([_str.split(',')[1].split(' ')[0] for _str in node.get_rssi().split(';')])
                    self.node_data[node.get_mac()]["location"].append((node._Node__f_lat_loc[node._Node__matched_row_offset],
                                                                       node._Node__f_lon_loc[node._Node__matched_row_offset]))

                    if node.get_mac() == TIMEKEEPER_MAC:
                        timestamps.append(node.get_time_stamp_in_s(node._Node__matched_row_offset))

                self.sec_offset_from_start += 1

            self.timestamps = [t - timestamps[0] for t in timestamps]

        else:
            print("Please provide a filename")

    def check_linearity_of_timesteps(self):
        """
        Checks whether the row-offset matching was done correctly
        The timestamp data is somewhat linear in its indices apart from occasional skips...
        """
        plt.plot(range(0, len(self.timestamps)), 'black', label="Linear relationship")
        plt.plot(self.timestamps, 'r*', linestyle='dashed', label=f"Timestamps recorded at {TIMEKEEPER_MAC}")
        plt.xlabel("Index"); plt.ylabel("Timestamp"); plt.legend()
        plt.show()

    def check_pathloss(self, tx_mac=None, rx_mac=None, video=False):
        """
        Fits the RSSI data against the log distance path loss model for RSS
        P_Rx = P_Tx - c - 10.plf.log(distance)
        NOTES:
            - Transmission powers are reported to be the same for all drones (27dBm)
            - c and plf (path loss factor) are constants which should be the same for all drones in the vicinity
            - Units of power are dBm, base of log can be anything
        :param tx_mac and rx_mac: only plot links transmitted/received from those nodes
        """
        distances = []
        rssi_vals = []
        if not video:
            cmap = cm.brg
            norm = Normalize(vmin=0, vmax=len(self.timestamps))
            colors = []  # Used to differentiate rssi values recorded at the start of the simulation vs. the end
            for t in range(len(self.timestamps)):
                for mac1 in self.node_macs:
                    if rx_mac and not rx_mac == mac1:
                        continue
                    nd1 = self.node_data[mac1]
                    phi1, lam1 = [np.deg2rad(nd1["location"][t][0]), np.deg2rad(nd1["location"][t][1])]
                    for i in range(len(nd1["neighbours"][t])):
                        if tx_mac and not tx_mac == nd1["neighbours"][t][i]:
                            continue
                        nd2 = self.node_data[nd1["neighbours"][t][i]]
                        phi2, lam2 = [np.deg2rad(nd2["location"][t][0]), np.deg2rad(nd2["location"][t][1])]
                        distances.append(get_geodesic_distance(phi1, lam1, phi2, lam2))
                        rssi_vals.append(float(nd1["rssi"][t][i]))
                        colors.append(cmap(norm(t)))
            log_distances = [np.log10(d) for d in distances]
            plt.scatter(log_distances, rssi_vals, s=8.0, alpha=0.5, c=colors)
            plt.xlabel(r"$log_{10}($distance (in m)$)$")
            plt.ylabel(r"RSSI (in $dBm$)")
            plt.show()
        else:
            pylab.ion()
            _ = pylab.get_current_fig_manager()

            if tx_mac or rx_mac:
                raise NotImplementedError

            plot_window = 15
            cmap = cm.YlOrRd
            norm = Normalize(vmin=0, vmax=plot_window-1)
            distances = [[] for _ in range(plot_window)]
            rssi_vals = [[] for _ in range(plot_window)]
            for t in range(len(self.timestamps)):
                pylab.clf()
                distances.pop(0)
                rssi_vals.pop(0)
                distances.append([])
                rssi_vals.append([])
                for mac1 in self.node_macs:
                    if rx_mac and not rx_mac == mac1:
                        continue
                    nd1 = self.node_data[mac1]
                    phi1, lam1 = [np.deg2rad(nd1["location"][t][0]), np.deg2rad(nd1["location"][t][1])]
                    for i in range(len(nd1["neighbours"][t])):
                        if tx_mac and not tx_mac == nd1["neighbours"][t][i]:
                            continue
                        nd2 = self.node_data[nd1["neighbours"][t][i]]
                        phi2, lam2 = [np.deg2rad(nd2["location"][t][0]), np.deg2rad(nd2["location"][t][1])]
                        distances[-1].append(get_geodesic_distance(phi1, lam1, phi2, lam2))
                        rssi_vals[-1].append(float(nd1["rssi"][t][i]))
                for i in range(plot_window):
                    plt.scatter([np.log10(d) for d in distances[-(1 + i)]], rssi_vals[-(1 + i)],
                                color=cmap(norm(plot_window-i-1)**4), alpha=norm(plot_window-i-1)**4, s=8.0)
                    plt.xlabel(r"$log_{10}($distance (in m)$)$")
                    plt.ylabel(r"RSSI (in $dBm$)")
                    pylab.xlim(0.8, 2.6)
                    pylab.ylim(-87, -30)
                plt.show()
                pause(0.0001)


def check_distance_calculations(center=(3, 3), sidelength=0.01, res=25):
    """ Visually verify that distance is being calculated correctly
    Overlays distance calculated using UTM coordinates with Euclidean distance vs. lat-lon with geodesic distance
    :param center: origin of distance calculation in lat lon (e.g., coordinates of the receiving drone)
    :param sidelength: side length of the grid corresponding to x and y axes
    :param res: resolution of the grid
    """
    lat1, lon1 = center
    dx, dy = np.meshgrid(np.linspace(-sidelength/2.0, sidelength/2.0, res),
                         np.linspace(-sidelength/2.0, sidelength/2.0, res))
    x = dx + np.ones([res, res])*lat1
    y = dy + np.ones([res, res])*lon1
    assert utm.from_latlon(x[0][0], y[0][0])[2] == utm.from_latlon(x[res-1][res-1], y[res-1][res-1])[2], \
        "The coordinates are in different UTM zones! Recall that 0 to 6 deg longitudes are in the same UTM zone."

    z_geo = np.zeros_like(dx)
    z_euc = np.zeros_like(dx)
    for i in range(res):
        for j in range(res):
            lat2, lon2 = (x[i][j], y[i][j])
            z_geo[i][j] = get_geodesic_distance(np.deg2rad(lat1), np.deg2rad(lon1), np.deg2rad(lat2), np.deg2rad(lon2))
            utm1 = utm.from_latlon(lat1, lon1)[0:2]
            utm2 = utm.from_latlon(lat2, lon2)[0:2]
            z_euc[i][j] = get_euclidean_distance(*utm1, *utm2)

    ax = plt.axes(projection='3d')
    _ = ax.plot_surface(x, y, z_geo, label="Geodesic Distance from LatLon", color='b', alpha=0.75)
    _ = ax.plot_surface(x, y, z_euc, label="Euclidean Distance from UTM", color='g', alpha=0.75)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Using geodesic/great circle distance (blue) \n vs. Euclidean distance (green)")
    plt.show()


def get_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_geodesic_distance(phi1, lam1, phi2, lam2):
    """ Get Geodesic distance from lat lon using the haversine function method
    :param phi1/2: latitude in radians
    :param lam1/2: longitude in radians
    :return: distance in meters
    """
    hav_theta = hav(phi2-phi1) + (1 - hav(phi1-phi2) - hav(phi1+phi2))*hav(lam2-lam1)
    return hav(hav_theta, inverse=True)*EARTH_RADIUS


def hav(theta, inverse=False):
    # The haversine function for calculating great circle distances
    if inverse:
        return np.arccos(1-2*theta)
    return (1-np.cos(theta))/2


if __name__ == '__main__':
    PATH = ""
    try:
        options, _ = getopt.getopt(sys.argv[1:], 'itp:m:a:', [])
        for opt, arg in options:
            if opt in '-p':
                PATH = arg

        if PATH == "":
            raise getopt.error("no path")
        file_list = os.listdir(PATH)
        for file in file_list:
            if file[0] == '.':
                file_list.remove(file)
                print(f"Hidden file '{file}' will be ignored.")

    except getopt.error:
        print("Provide a path to csv files...")
        sys.exit(2)

    eval = NodeNetworkEvaluator(False, [], False)
    eval.collect_data(PATH, file_list)

    # Uncomment these to verify the corresponding functionality...
    # runner.check_linearity_of_timesteps()
    # check_distance_calculations()
    eval.check_pathloss(video=True)
