#!/usr/bin/env bash

echo="echo -e"

trap ctrl_c INT

function ctrl_c() {
    $echo "** Trapped CTRL-C"
    cleanup
    exit
}

declare -a pids
declare -a keys

function cleanup {
    if [ ${#pids[@]} -gt 0 ]
    then
	$echo "existing pids are ${pids[@]}"
	for pid in "${pids[@]}"
	do
	    $echo "killing pid $pid"
	    kill -9 $pid
	done
    else
	$echo "we do not have any existing processes"
    fi
    
    if [ ${#keys[@]} -gt 0 ]
    then
	$echo "existing ring buffer keys are ${keys[@]}"
       for key in "${keys[@]}"
       do
    	   $echo "removing ring buffer $key"
    	   dada_db -k $key -d
       done
    else
	$echo "we do not have any existing ring buffers"
    fi
}

$echo "This is a pipeline for beamforming\n"

# There will be two ring buffers
# The first one is used to receive raw data
# The second one is used to receive beamformed data

# The first ring buffer need two readers if we also want to save the raw data
# The second ring buffer has only one reader

# some path of the code
project_root=/home/wangyu/code/PAF_pipeline
hdr_root=$project_root/header
beamform_command=$project_root/build/pipeline/pipeline_dada_beamform
udp_command=$project_root/build/udp/udp2db

$echo "Setting up ring buffers"
key_raw=a000
key_bmf=b000
pkt_dtsz=4096
n_antenna=10
pkt_per_block=256
n_chan=256
sampsz=2
n_average=8
n_beam=180
timestep_per_pkt=$((pkt_dtsz/n_chan/sampsz))
timestep_per_block=$((timestep_per_pkt*pkt_per_block))
initsampsz=4


# it will be more flexible if we put equation here to calculate buffer size with some basic configrations
bufsz_raw=$((pkt_dtsz*pkt_per_block*n_antenna)) # buffer block size to hold raw data in bytes, change it to real value later
bufsz_bmf=$((timestep_per_block*initsampsz*n_beam*n_chan/n_average)) # buffer block size to hold beamformed data in bytes, change it to real value later

$echo "key_raw is $key_raw, and bufsz_raw is $bufsz_raw"
$echo "key_bmf is $key_bmf, and bufsz_bmf is $bufsz_bmf\n"

numa=0 # numa node to use 

$echo "Creating ring buffers"
dada_raw="dada_db -k $key_raw -b $bufsz_raw -p -w -c $numa -r 2"  #assign memory from NUMA node  [default: all nodes]
dada_bmf="dada_db -k $key_bmf -b $bufsz_bmf -p -w -c $numa"

$dada_raw & # should be unblock 
$dada_bmf & # should be unblock

sleep 1s # to wait all buffers are created 
$echo "\nSetting file writers if you want save the dada files"
# different type of files should go to different directories
dir_raw=/home/wangyu/data/data_raw/ # change it to real value later
dir_bmf=/home/wangyu/data/data_bmf/ # change it to real value later

# 设定写入数据的地址和数据来源ringbuffer
writer_raw="dada_dbdisk -D $dir_raw -k $key_raw -W" 
writer_bmf="dada_dbdisk -D $dir_bmf -k $key_bmf -W"

$echo "writer_raw $writer_raw"
$echo "writer_bmf $writer_bmf\n"

$writer_raw & # should be unblock 
$writer_bmf & # should be unblock

# start beamform pipeline
$echo "Start beamform process"
process="$beamform_command -i $key_raw -o $key_bmf -g 1" # need to add other configurations as well
$echo "$process\n"
$process & # should be unblock

# set up udp2db
nblock=10
nsecond=10
hdr_fname=$hdr_root/paf_test.header

# Start udp2db
$echo "Start udp2db process"
udp_process="$udp_command -k $key_raw -n $nblock -N $nsecond -s 1"
$echo "$udp_process\n"
$udp_process

sleep 1s
$echo "done udp2db setup\n"
cleanup
