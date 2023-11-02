#!/usr/bin/env bash

echo="echo -e"

$echo "This is a pipeline for beamforming at NAOC\n"

# There will be three ring buffers
# The first one is used to receive raw data
# The second one is used to receive beamformed data
# The last one is used to receive zoom fft data

# The first ring buffer need two readers as we need to read its data to GPU and write its data to disk
# The second ring buffer has only one reader
# The last ring buffer has only one reader as well

$echo "Setting up ring buffers"

key_raw=a000
key_bmf=b000

$echo "key_raw is $key_raw"
$echo "key_bmf is $key_bmf"

# it will be more flexible if we put equation here to calculate buffer size with some basic configrations
bufsz_raw=25165824 # buffer block size to hold raw data in bytes, change it to real value later
bufsz_bmf=20971 # buffer block size to hold beamformed data in bytes, change it to real value later

$echo "bufsz_raw is $bufsz_raw"
$echo "bufsz_bmf is $bufsz_bmf"

numa=0 # numa node to use 
$echo "numa is $numa\n"

$echo "Creating ring buffers"
dada_raw="dada_db -k $key_raw -b $bufsz_raw -p -w -c $numa -r 2"  #assign memory from NUMA node  [default: all nodes]
dada_bmf="dada_db -k $key_bmf -b $bufsz_bmf -p -w -c $numa"

$echo "dada_raw is $dada_raw"
$echo "dada_bmf is $dada_bmf \n"

$dada_raw & # should be unblock 
$dada_bmf & # should be unblock

sleep 1s # to wait all buffers are created 
$echo "Setting file writers"
# different type of files should go to different directories
dir_raw=/home/hero/data/data_raw/ # change it to real value later
dir_bmf=/home/hero/data/data_bmf/ # change it to real value later

$echo "dir_raw $dir_raw"
$echo "dir_bmf $dir_bmf"

# 设定写入数据的地址和数据来源ringbuffer
writer_raw="dada_dbdisk -D $dir_raw -k $key_raw -W" 
writer_bmf="dada_dbdisk -D $dir_bmf -k $key_bmf -W"

$echo "writer_raw $writer_raw"
$echo "writer_bmf $writer_bmf\n"

$writer_raw & # should be unblock 
$writer_bmf & # should be unblock

# now gpu pipeline
$echo "Starting process"
process="../build/pipeline/pipeline_dada_beamform -i $key_raw -o $key_bmf -g 0" # need to add other configurations as well
$echo "process $process\n"
$process & # should be unblock

# now udp2db
udp2db="bash ./udp2db.sh"
$echo "udp2db $udp2db\n"
$udp2db

sleep 1s # to wait all process finishes
$echo "Killing dada_dbdisk"
writer_d="pkill -9 dada_dbdisk" #
$echo "writer_d $writer_d\n"
$writer_d # we should not block here 

$echo "\nRemoving ring buffers"
dada_raw_d="dada_db -k $key_raw -d"
dada_bmf_d="dada_db -k $key_bmf -d"

$echo "dada_raw_d is $dada_raw_d"
$echo "dada_bmf_d is $dada_bmf_d\n"

$dada_raw_d
$dada_bmf_d
