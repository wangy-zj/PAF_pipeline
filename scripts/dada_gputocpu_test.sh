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
    	   ipcrm -a
           #dada_db -k $key -d
       done
       $echo "removed all ring buffers"
    else
	$echo "we do not have any existing ring buffers"
    fi
}


WORK_ROOT=/home/wangyu
project_root=$WORK_ROOT/PAF_pipeline
hdr_root=$project_root/header/512MHz_beamform_4096B.header
dada_gputocpu=$project_root/build/pipeline/dada_gputocpu_test
dada_cputogpu=$project_root/build/pipeline/dada_cputogpu_test

dada_dtsz=$((2000*1024*1024))
key_input=a000
key_gpu=b000
key_output=c000
block_num=8

# 输入ringbuffer
dada_db -k $key_input -n $block_num -b $dada_dtsz -p -l -w&
pids+=(`echo $! `)
keys+=(`echo $key_input `)
sleep 1s 
# GPU ringbuffer
dada_db -k $key_gpu -n $block_num -b $dada_dtsz -g 0 -p -l -w&
pids+=(`echo $! `)
keys+=(`echo $key_gpu `)
sleep 1s 

dada_db -k $key_output -n $block_num -b $dada_dtsz -p -l -w&
pids+=(`echo $! `)
keys+=(`echo $key_output `)
sleep 1s 
$echo "created all ring buffers\n"


# 清除ringbuffer
dada_dbnull -k $key_output&
pids+=(`echo $! `)
sleep 1s
$echo "nulling out GPU ring buffer\n"

# 开启传输
$dada_gputocpu -g $key_gpu -o $key_output&
pids+=(`echo $! `)
sleep 1s 
$dada_cputogpu -i $key_input -g $key_gpu&
pids+=(`echo $! `)
sleep 1s 
$echo "started transfer\n"

dada_junkdb -g -R 7000 -z -k $key_input -t 60 $hdr_root
pids+=(`echo $! `)
sleep 1s 
$echo "junked CPU ring buffer\n"

cleanup
