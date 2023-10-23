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
nstream_gpu=24
npkt=2048
numa=0
key=a000
bufsz=$(( pkt_dtsz*nstream_gpu*npkt ))
$echo "pkt_dtsz is:    $pkt_dtsz"
$echo "nstream_gpu is: $nstream_gpu"
$echo "npkt is:        $npkt"
$echo "numa is:        $numa"
$echo "DADA key is:    $key"
$echo "bufsz is:       $bufsz\n"

# create PSRDADA ring buffer
dada_db -k $key -b $bufsz -p -l -c $numa -w &
pids+=(`echo $! `)
keys+=(`echo $key `)
sleep 1s # just to make sure that all ring buffers are created
$echo "created all ring buffers\n"

# setup data consumers
#dada_dbnull -k $key -z &
dada_dbdisk -k $key -D . -W &
pids+=(`echo $! `)
$echo "had the data consumer up\n"

# setup tests
hdr_fname=$hdr_root/paf_test.header
nblock=1
nsecond=10
freq=1420

$echo "hdr_fname is: $hdr_fname"
$echo "nblock is:    $nblock"
$echo "nsecond is:   $nsecond"
$echo "freq is:      $freq\n"

# Start udp2db
$udp_command -f $hdr_fname -F $freq -n $nblock -N $nsecond -k $key
sleep 1s
$echo "done udp2db setup\n"
cleanup
