import getopt
import os
import sys

import utm

from ftl_player import NodeNetwork

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True

TIMEKEEPER_MAC = '00:30:1a:4f:5b:2f'
EARTH_RADIUS = 6371000  # Meters


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
            print("Please give filename")

    def check_linearity_of_timesteps(self):
        """
        Checks whether the row-offset matching was done correctly. The timestamp data is somewhat linear in its indices
        apart from occasional skips...
        """
        plt.plot(range(0, len(self.timestamps)), 'black', label="Linear relationship")
        plt.plot(self.timestamps, 'r*', linestyle='dashed', label=f"Timestamps recorded at {TIMEKEEPER_MAC}")
        plt.xlabel("Index"); plt.ylabel("Timestamp"); plt.legend()
        plt.show()

    def check_pathloss(self):
        """
        Fits the RSSI data against the log distance path loss model for RSS
        P_Rx = P_Tx - c - 10.plf.log(distance)
        NOTES:
            - Transmission powers are reported to be the same for all drones (27dBm)
            - c and plf (path loss factor) are constants which should be the same for all drones in the vicinity
            - Units of power are dBm, base of log can be anything
        """
        distances = []
        rssi_vals = []
        for t in range(len(self.timestamps)):
            for mac1 in self.node_macs:
                nd1 = self.node_data[mac1]
                phi1, lam1 = [np.deg2rad(nd1["location"][t][0]), np.deg2rad(nd1["location"][t][1])]
                for i in range(len(nd1["neighbours"][t])):
                    nd2 = self.node_data[nd1["neighbours"][t][i]]
                    phi2, lam2 = [np.deg2rad(nd2["location"][t][0]), np.deg2rad(nd2["location"][t][1])]
                    distances.append(get_geodesic_distance(phi1, lam1, phi2, lam2))
                    rssi_vals.append(float(nd1["rssi"][t][i]))
        log_distances = [np.log10(d) for d in distances]
        plt.scatter(log_distances, rssi_vals, s=1.0, c='black')
        plt.xlabel(r"$log_{10}($distance (in m)$)$")
        plt.ylabel(r"RSSI (in $dBm$)")
        plt.show()


def check_distance_calculations(center=(3, 3), sidelength=0.1, res=120):
    """
    Overlays distance calculated using UTM coordinates with Euclidean distance
    vs. lat-lon coordinates with geodesic distance
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

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf1 = ax.plot_surface(x, y, z_geo, label="Geodesic Distance from LatLon", color='b', alpha=0.75)
    surf2 = ax.plot_surface(x, y, z_euc, label="Euclidean Distance from UTM", color='g', alpha=0.75)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Using geodesic/great circle distance (blue) \n vs. \n Euclidean distance (green) within the same UTM zone")
    plt.show()


def get_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_geodesic_distance(phi1, lam1, phi2, lam2):
    """ Get Geodesic distance from lat lon using the haversine function method
    :param phi1: latitude in radians
    :param lam2: longitude in radians
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

    runner = NodeNetworkEvaluator(False, [], False)
    runner.collect_data(PATH, file_list)
    # check_distance_calculations()
    # runner.check_linearity_of_timesteps()
    runner.check_pathloss()


