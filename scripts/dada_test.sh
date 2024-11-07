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


WORK_ROOT=/home/hero/code
project_root=$WORK_ROOT/PAF_pipeline
hdr_root=$project_root/header
dada_command=$project_root/build/pipeline/dada_speed_test

dada_dtsz=1000000
key_cpu=a000
key_gpu=b000

# 输入ringbuffer
dada_db -k $key_cpu -n 8 -b $dada_dtsz -w&
$echo "dada_db -k $key_cpu -n 8 -b $dada_dtsz -w&"
pids+=(`echo $! `)
keys+=(`echo $key_cpu `)
sleep 1s 
# GPU ringbuffer
dada_db -k $key_gpu -n 8 -b $dada_dtsz -g 0 -w&
pids+=(`echo $! `)
keys+=(`echo $key_gpu `)
sleep 1s 
$echo "created all ring buffers\n"

# 输出ringbuffer
#dada_db -k a002 -b $dada_dtsz -p -w&

# 清除ringbuffer
dada_dbnull -k $key_gpu&
pids+=(`echo $! `)
sleep 1s

# 开启传输
$dada_command -i $key_cpu -o $key_gpu -g 0&
pids+=(`echo $! `)
sleep 1s 

dada_junkdb -g -r 1000 -z $key_cpu
pids+=(`echo $! `)
sleep 1s 

cleanup

