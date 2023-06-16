import json
import numpy

with open("data/rssi_measurement_set1.json", "rb") as f:
    data = json.load(f)

from mcap.reader import make_reader
from rosbags.highlevel.anyreader import deserialize_cdr

MCAP_FILENAME = "data/rssi_measurement_set1_mcap/sad03_mission1.mcap"

data["rssi"] = []
data["rssi_timestamps"] = []
with open(MCAP_FILENAME, "rb") as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages("/sad03/mesh_visual"):
        data["rssi"].append(
            json.loads(deserialize_cdr(message.data, schema.name).data)
            )
        data["rssi_timestamps"].append(message.log_time)

data["rssi_positions"] = []
position_index = 0
rssi_index = 0
with open(MCAP_FILENAME, "rb") as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages('/sad03/drone/global_position'):
        if data["rssi_timestamps"][rssi_index] < message.log_time:
            data["rssi_positions"].append((data["x"][position_index], 
                                           data["y"][position_index], 
                                           data["z"][position_index]))
            rssi_index += 1
            if rssi_index >= len(data["rssi"]):
                break
        position_index += 1
        
# TX Mac: 04:f0:21:9b:4b:78
# 

data["distances"] = []
origin_xyz = numpy.array([[data['x'][0]], [data['y'][0]], [data['z'][0]]])
for rssi_xyz in zip(data["rssi"], data["rssi_positions"]):
    receiver = numpy.array([rssi_xyz[1][0], rssi_xyz[1][1], rssi_xyz[1][2]])
    distance = numpy.linalg.norm(receiver - origin_xyz)
    data["distances"].append(distance)

# for end_time in [50, 100, 150, 200, 250, 300, 360]:
#     plot.plot([numpy.log(d) for d in data["distances"][:end_time]], 
#             [float(r[1]['devices'][0]['nhr'][0]) for r in data["rssi"][:end_time]], "b*-")
#     plot.xlabel("Log(Distance)")
#     plot.ylabel("RSSI (in dBm)")
#     plot.tight_layout()
#     plot.show()

averaged_rssi = [0.25*(float(r[1]['devices'][0]['nhr'][0]) 
           + float(r[1]['devices'][0]['nhr'][1]) 
           + float(r[1]['devices'][0]['nhr'][2]) 
           + float(r[1]['devices'][0]['nhr'][3])) for r in data["rssi"]]

x_vals = []
y_vals = []
for r,d in zip(averaged_rssi, data["distances"]):
    if numpy.log(d) < 3.5:
        y_vals.append(r)
        x_vals.append(numpy.log(d))

H = numpy.array([[1, val] for val in x_vals])
line_fit = numpy.linalg.inv(H.T @ H) @ H.T @ numpy.array(y_vals)

plot.plot([numpy.log(d) for d in data["distances"]], 
          [float(r[1]['devices'][0]['nhr'][0]) for r in data["rssi"]], "b-", label="DATA['nhr'][0]", linewidth=0.5, markersize=2, alpha=0.4)
plot.plot([numpy.log(d) for d in data["distances"]], 
          [float(r[1]['devices'][0]['nhr'][1]) for r in data["rssi"]], "c-", label="DATA['nhr'][1]", linewidth=0.5, markersize=2, alpha=0.4)
plot.plot([numpy.log(d) for d in data["distances"]], 
          [float(r[1]['devices'][0]['nhr'][2]) for r in data["rssi"]], "g-", label="DATA['nhr'][2]", linewidth=0.5, markersize=2, alpha=0.4)
plot.plot([numpy.log(d) for d in data["distances"]], 
          [float(r[1]['devices'][0]['nhr'][3]) for r in data["rssi"]], "r-", label="DATA['nhr'][3]", linewidth=0.5, markersize=2, alpha=0.4)

plot.plot([1.5, 4.1], [line_fit[0] + line_fit[1]*1.5, line_fit[0] + line_fit[1]*4.1], "b--", label="Line Fit", linewidth=0.75)
plot.plot([numpy.log(d) for d in data["distances"]], averaged_rssi, "k*-", label="Averaged Value]", linewidth=0.5, markersize=2)
plot.xlabel("Log(Distance)")
plot.ylabel("RSSI (in dBm)")
plot.legend()
plot.tight_layout()
plot.show()


xs = numpy.linspace(1.5, 4.1, 100)
ys = []

for x in xs:
    ys.append(numpy.exp(x + 0.25) - numpy.exp(x))
plot.plot(xs, ys)
plot.show()

"""
---------------------------
3D Plot of Drone Trajectory
---------------------------
"""

import matplotlib.pyplot as plot
from mpl_toolkits import mplot3d

ax = plot.axes(projection='3d')
ax.plot3D(data['x'], data['y'], [val*(-1) for val in data['z']])
# Note: The coordinate system may be right/left handed, not sure. The z-axis of the dataset points into the ground.
plot.show()

"""
---------------------------- ---------------------------- ----------------------------
Below are examples of how the 'mesh_visual' data is organized. For documentation/reference purposes only
A "batadv-vis server" is used in each node to log the mesh topology and RSSI information
---------------------------- ---------------------------- ----------------------------
"""

def ExampleMeshVisualMessage(**kwargs):
    print("This is a dummy function which I use to trigger the syntax highlighting in my IDE :)")
    pass

_ = ExampleMeshVisualMessage(channel_id=7, 
            log_time=1678711802621392000, 
            data=b'\x00\x01\x00\x00o\x02\x00\x00[{"source_version":"debian-2022.0-1","algorithm":4,"vis":[{"primary":"00:30:1a:4f:5b:0b","neighbors":[{"router":"00:30:1a:4f:5b:0b","neighbor":"04:f0:21:9b:4b:78","metric":"1.008"}]},{"primary":"04:f0:21:9b:4b:78","neighbors":[{"router":"04:f0:21:9b:4b:78","neighbor":"00:30:1a:4f:5b:0b","metric":"1.016"}]}]},{"ts":"2023-03-13 12:50:02.522485","status":"MESH","my_mac":"04:f0:21:9b:4b:78","noise":"-95","freq":"5745","txpower":"30.00","country":"US","hw":"Wi-Fi","devices":[{"a":"1","o":"00:30:1a:4f:5b:0b","ls":"0.224","q":"251","nh":"00:30:1a:4f:5b:0b","or":["-54","-62","-63","-56"],"nhr":["-54","-62","-63","-56"]}]}]\x00\x00', 
            publish_time=1678711802621392000, 
            sequence=2)

example_mesh_visual_data = [
    # batadv-vis related information
    {"source_version": "debian-2022.0-1",
     "algorithm": 4,
     "vis" :[
         # I am my neighbor's neighbor (i.e., my neighbor sees me). 
         # 'metric' = the number of times a packet needs to be transmitted to get it delivered. 1.00 is the best.
         {"primary" :"00:30:1a:4f:5b:0b",  # My neighbor
          "neighbors" :[{"router":"00:30:1a:4f:5b:0b", "neighbor":"04:f0:21:9b:4b:78", "metric":"1.008"}]},
         {"primary" :"04:f0:21:9b:4b:78",  # This is me
          "neighbors" :[{"router":"04:f0:21:9b:4b:78", "neighbor":"00:30:1a:4f:5b:0b", "metric":"1.016"}]}
         ]},

    {"ts": "2023-03-13 12:50:02.522485",
     "status": "MESH",  # Takes a value in AP, NO_CONFIG, MESH, or ERROR
     "my_mac": "04:f0:21:9b:4b:78",  # MAC where the report is received
     "noise": "-95",  # Background noise level, -95dBm or bigger starts eating link budget
     "freq": "5745",  # Used WiFi frequency in MHz
     "txpower": "30.00",  # TX (transmission) power in dBm
     "country": "US",
     "hw": "Wi-Fi",
     "devices": [
         # List of neighbors
         {"a": "1",  # 1 => "active towards originator" not sure what that means.
          "o": "00:30:1a:4f:5b:0b",  # This is the receiver/originator of the IP packet
          "ls": "0.224",  # Last seen in network (in seconds)
          "q": "251",  #  Quality value (0 to 255). No. of hops affects this as well as 'metric'
          "nh": "00:30:1a:4f:5b:0b",  # Next hop to reach receiver/originator
          "or": ["-54","-62","-63","-56"],  # RSSI values corresponding to the originator/receiver (given by 'o')
          "nhr": ["-54","-62","-63","-56"]}  # RSSI values corresp. to the next hop neighbor (given by 'nh')
         ]}
    ]

"""
all_data = {1: Channel(id=1, topic='/fmu/out/SensorCombined', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=1),
            2: Channel(id=2, topic='/fmu/out/VehicleGpsPosition', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=2),
            3: Channel(id=3, topic='/sad03/drone/global_position', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=3),
            4: Channel(id=4, topic='/sad03/drone/home_position', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=4),
            5: Channel(id=5, topic='/sad03/drone/local_odom', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=5),
            6: Channel(id=6, topic='/sad03/drone/local_position', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=6),
            7: Channel(id=7, topic='/sad03/mesh_visual', message_encoding='cdr', metadata={'offered_qos_profiles': ''}, schema_id=7)}
"""