#!/bin/bash
gateway=$1
meshcom_path=$(pwd | awk -F 'mesh_com' '{print $1 FS "/"}')
sc_path=$(pwd | awk -F 'sc-mesh-secure-deployment' '{print $1 FS "/"}')


route add default gw $gateway bat0
cp $meshcom_path/common/scripts/mesh-default-gw.sh /usr/sbin/.
chmod 744 /usr/sbin/mesh-default-gw.sh
cp $sc_path/services/initd/S91defaultroute /etc/init.d/.
chmod 777 /etc/init.d/S91defaultroute
/etc/init.d/S91defaultroute start $gateway
