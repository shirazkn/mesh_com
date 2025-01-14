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
import matplotlib as mpl

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

            self.node_data = {node.get_mac(): {"neighbours": [], "rssi": [], "neighbour_txmcs": [], "location": [], "altitude": []}
                              for node in self.network_nodes}
            timestamps = []
            for _ in range(total_time):
                for node in self.network_nodes:
                    node.update_row_offset_from_seconds_offset(self.sec_offset_from_start)

                for node in self.network_nodes:
                    self.node_data[node.get_mac()]["neighbours"].append([_str.split(',')[0] for _str in node.get_rssi().split(';')])
                    self.node_data[node.get_mac()]["rssi"].append([_str.split(',')[1].split(' ')[0] for _str in node.get_rssi().split(';')])

                    self.node_data[node.get_mac()]["neighbour_txmcs"].append([])
                    for neighbor_mac in self.node_data[node.get_mac()]["neighbours"][-1]:
                        neighbor_mcs = self._get_txmcs(from_mac=node.my_mac, to_mac=neighbor_mac)
                        self.node_data[node.get_mac()]["neighbour_txmcs"][-1].append(neighbor_mcs)

                    self.node_data[node.get_mac()]["location"].append((node._Node__f_lat_loc[node._Node__matched_row_offset],
                                                                       node._Node__f_lon_loc[node._Node__matched_row_offset]))
                    self.node_data[node.get_mac()]["altitude"].append(node._Node__f_altitude[node._Node__matched_row_offset])

                    if node.get_mac() == TIMEKEEPER_MAC:
                        timestamps.append(node.get_time_stamp_in_s(node._Node__matched_row_offset))

                self.sec_offset_from_start += 1

            self.timestamps = [t - timestamps[0] for t in timestamps]
            for node in self.network_nodes:
                pad_zeroes(self.node_data[node.get_mac()]["altitude"])

        else:
            print("Please provide a filename")

    def get_node_from_mac(self, mac):
        for node in self.network_nodes:
            if node.my_mac == mac:
                return node
        raise ValueError("There is no node with the specified MAC!")

    def _get_txmcs(self, from_mac, to_mac):
        to_node = self.get_node_from_mac(to_mac)
        from_node = self.get_node_from_mac(from_mac)
        txmcs = None
        for step in [0, -1, -2, -3, 1, 2, 3]:
            if txmcs:
                break
            for mac_mcs in from_node._Node__f_txmcs[from_node._Node__matched_row_offset - step].split(';'):
                if mac_mcs.split(',')[0] == to_node.my_mac:
                    txmcs = mac_mcs.split(',')[1]
                    break
        if not txmcs:
            raise ValueError(f"Could not find TX MCS at {self.sec_offset_from_start} seconds.")
        return int(txmcs)

    def check_linearity_of_timesteps(self):
        """
        Checks whether the row-offset matching was done correctly
        The timestamp data is somewhat linear in its indices apart from occasional skips...
        """
        plt.plot(range(0, len(self.timestamps)), 'black', label="Linear relationship")
        plt.plot(self.timestamps, 'r*', linestyle='dashed', label=f"Timestamps recorded at {TIMEKEEPER_MAC}")
        plt.xlabel("Index"); plt.ylabel("Timestamp"); plt.legend()
        plt.show()

    def check_pathloss(self, tx_mac=None, rx_mac=None, tx_mcs=None, video=False):
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
            cmap = mpl.colormaps["gist_ncar"]
            # norm = Normalize(vmin=0, vmax=len(self.timestamps))
            norm = Normalize(vmin=0, vmax=18)
            colors = []  # Used to differentiate rssi values recorded at the start of the simulation vs. the end
            for t in range(len(self.timestamps)):
                for mac1 in self.node_macs:
                    if rx_mac and not rx_mac == mac1:
                        continue
                    nd1 = self.node_data[mac1]
                    phi1, lam1, alt1 = [np.deg2rad(nd1["location"][t][0]), np.deg2rad(nd1["location"][t][1]), nd1["altitude"][t]]
                    for i in range(len(nd1["neighbours"][t])):
                        if tx_mac and not tx_mac == nd1["neighbours"][t][i]:
                            continue
                        if tx_mcs and not tx_mcs == nd1["neighbour_txmcs"][t][i]:
                            continue
                        nd2 = self.node_data[nd1["neighbours"][t][i]]
                        phi2, lam2, alt2 = [np.deg2rad(nd2["location"][t][0]), np.deg2rad(nd2["location"][t][1]), nd2["altitude"][t]]
                        h_dist = get_geodesic_distance(phi1, lam1, phi2, lam2)
                        distances.append(get_euclidean_distance(h_dist, alt2-alt1))
                        rssi_vals.append(float(nd1["rssi"][t][i]))
                        colors.append(cmap(norm(nd1["neighbour_txmcs"][t][i])))
            log_distances = [np.log10(d) for d in distances]
            plt.scatter(log_distances, rssi_vals, s=5.0, alpha=0.35, c=colors)
            plt.xlabel(r"$log_{10}($distance (in m)$)$")
            plt.ylabel(r"RSSI (in $dBm$)")
            title = ""
            if tx_mac:
                title += f"TX Node: {tx_mac} \n"
            if rx_mac:
                title += f"RX Node: {rx_mac}"
            plt.title(title)
            pylab.xlim(0.8, 2.5)
            pylab.ylim(-86, -30)
            plt.show()

        else:
            pylab.ion()
            fig = pylab.get_current_fig_manager()
            fig.canvas.mpl_connect('close_event', self._NodeNetwork__on_close)
            fig.canvas.mpl_connect('key_press_event', self._NodeNetwork__on_key_press_event)

            plot_window = 4
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
                    phi1, lam1, alt1 = [np.deg2rad(nd1["location"][t][0]), np.deg2rad(nd1["location"][t][1]), nd1["altitude"][t]]
                    for i in range(len(nd1["neighbours"][t])):
                        if tx_mac and not tx_mac == nd1["neighbours"][t][i]:
                            continue
                        nd2 = self.node_data[nd1["neighbours"][t][i]]
                        phi2, lam2, alt2 = [np.deg2rad(nd2["location"][t][0]), np.deg2rad(nd2["location"][t][1]), nd2["altitude"][t]]
                        h_dist = get_geodesic_distance(phi1, lam1, phi2, lam2)
                        distances[-1].append(get_euclidean_distance(h_dist, alt2-alt1))
                        rssi_vals[-1].append(float(nd1["rssi"][t][i]))
                for i in range(plot_window):
                    plt.scatter([np.log10(d) for d in distances[-(plot_window)+i]], rssi_vals[-(plot_window)+i],
                                color=cmap(norm(i)**4), alpha=norm(i), s=8.0)
                    plt.xlabel(r"$log_{10}($distance (in m)$)$")
                    plt.ylabel(r"RSSI (in $dBm$)")

                # plt.gca().set_xticks(np.linspace(0, 3.0, 61), minor=True)
                # plt.gca().set_yticks(np.linspace(-90, -30, 61), minor=True)
                # plt.grid(which='minor', alpha=0.2)
                # plt.grid(which='major', alpha=0.5)
                pylab.xlim(0.8, 2.6)
                pylab.ylim(-87, -30)
                plt.show()
                pause(0.00001)


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


def get_euclidean_distance(x1, y1, x2=0.0, y2=0.0):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def pad_zeroes(array, thresh=1e-2):
    # Make sure first entry is non-zero
    i = 1
    while array[0] < thresh:
        array[0] = array[i]
        i += 1

    # Pad remaining entries
    for i in range(1, len(array)):
        if array[i] < thresh:
            array[i] = array[i-1]


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
            PATH = "./data"
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
    eval.check_pathloss(video=False)

