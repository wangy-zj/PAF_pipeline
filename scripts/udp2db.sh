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

# setup command lines
WORK_ROOT=/home/hero/code
project_root=$WORK_ROOT/PAF_pipeline
hdr_root=$project_root/header
udp_command=$project_root/build/udp/udp2db

$echo "project_root is: $project_root"
$echo "hdr_root is:     $hdr_root"
$echo "udp_command is:  $udp_command\n"

# setup dada buffer
pkt_dtsz=4096
nantenna=10
npkt=256
key_cpu=a000
key_gpu=a001
bufsz=$(( pkt_dtsz*nantenna*npkt ))
$echo "pkt_dtsz is:    $pkt_dtsz"
$echo "nantenna is:    $nantenna"
$echo "npkt is:        $npkt"
$echo "CPU DADA key is:    $key_cpu"
$echo "GPU DADA key is:    $key_gpu"

# create PSRDADA ring buffer
dada_db -k $key_cpu  -b $bufsz -n 16 -w &
dada_db -k $key_gpu  -b $bufsz -n 16 -g 0 -w &
pids+=(`echo $! `)
keys+=(`echo $key_cpu `)
sleep 1s # just to make sure that all ring buffers are created
$echo "created all ring buffers\n"

# setup data consumers
#dada_dbnull -k $key -z &
dir_raw=/home/hero/data/data_raw
#dada_dbdisk -b 1 -k $key_cpu -D $dir_raw -o -z -W &
#pids+=(`echo $! `)
#$echo "had the data consumer up\n"

# setup tests
hdr_fname=$hdr_root/paf_test.header
nsecond_report=2
nsecond=10
nblocksave=40

$echo "nblock is:    $nblock"
$echo "nsecond is:   $nsecond"

dada_dbcopydb -z -v $key_cpu $key_gpu &
pids+=(`echo $! `)
$echo "copy the data from CPU to GPU\n"

# Start udp2db
$udp_command -f $hdr_fname -n $nsecond_report -N $nsecond -k $key_cpu -s $nblocksave
sleep 1s
$echo "done udp2db setup\n"
cleanup
