# udp-pipeline

We DO NOT need to transport data before do time accumulation

# To do
1. apply new utilities to this project? done
2. 

# Test the pipeline with junk data

## performance test
1. Create input and output ring buffer
```
dada_db -k a000 -b 536870912 -p -w
dada_db -k b000 -b 16388 -p -w 
```

2. Run `dada_dbnull` to clean output ring buffer
```
dada_dbnull -k b000
```

3. Run `dada_junkdb` to fill data into input ring buffer

```
dada_junkdb -k a000 -n header/500MHz_1ant1pol.header -r 6400 -t 1
```

4. Run `pipeline_dada_1ant1pol` to process data
```
pipeline_dada_1ant1pol -i a000 -o b000 -g 0 -n 128
```

The pipeline runs fine, it process ~540 milliseconds data in100.7 milliseconds and half of the time is for memocy copy from host to device. 
```
DEBUG: gpu = 0
DEBUG: nthread = 128
DEBUG: input_key = a000
DEBUG: output_key = b000
gpuDeviceInit() CUDA Device [0]: "Ampere
Asked for GPU 0, got GPU 0
GPU name is NVIDIA GeForce RTX 3080
PROCESS_INFO:	We have input HDU locked
PROCESS_INFO:	We have input HDU setup
PROCESS_INFO:	We have output HDU locked
PROCESS_INFO:	We have output HDU setup
HERE
DEBUG: gpu = 0
DEBUG: npkt = 65536
DEBUG: nfft = 65536
DEBUG: nchan_fine = 4097
DEBUG: nthread = 128
DEBUG: nsamp_packed = 536870912
DEBUG: nsamp_fft = 268500992
PROCESS_INFO:	input buffer block size is 536870912 bytes, output buffer block size is 16388 bytes
PROCESS_INFO:	We have input buffer block size checked
PROCESS_INFO:	We have output buffer block size checked
PROCESS_INFO:	 device input buffer size is 8192d bytes
GPU free memory is 2934.1, total is 10016.8 MBbytes
We are at 0 block
Memory copy from host to device of 0 block done
We are at 1 block
Memory copy from host to device of 1 block done
We are at 2 block
Memory copy from host to device of 2 block done
We are at 3 block
Memory copy from host to device of 3 block done
We are at 4 block
Memory copy from host to device of 4 block done
We are at 5 block
Memory copy from host to device of 5 block done
We are at 6 block
Memory copy from host to device of 6 block done
We are at 7 block
Memory copy from host to device of 7 block done
We are at 8 block
Memory copy from host to device of 8 block done
We are at 9 block
Memory copy from host to device of 9 block done
We are at 10 block
Memory copy from host to device of 10 block done
We are at 11 block
Memory copy from host to device of 11 block done
We are at 12 block
Memory copy from host to device of 12 block done
pipeline   100.729752 milliseconds, pipline with memory transfer averaged with 13 blocks
pipeline   55.547119 milliseconds, memory transfer h2d averaged with 13 blocks
pipeline   0.000000 milliseconds, memory transfer d2h averaged with 13 blocks
available  536.870912 milliseconds, available time for pipeline
```

## memory check
No memory isse wit`cuda-memcheck`
```
super@super007:~/udp-pipeline/build$ cuda-memcheck ./pipeline_dada_1ant1pol -i a000 -o b000 -g 0 -n 128
========= CUDA-MEMCHECK
========= This tool is deprecated and will be removed in a future release of the CUDA toolkit
========= Please use the compute-sanitizer tool as a drop-in replacement
DEBUG: gpu = 0
DEBUG: nthread = 128
DEBUG: input_key = a000
DEBUG: output_key = b000
gpuDeviceInit() CUDA Device [0]: "Ampere
Asked for GPU 0, got GPU 0
GPU name is NVIDIA GeForce RTX 3080
PROCESS_INFO:	We have input HDU locked
PROCESS_INFO:	We have input HDU setup
PROCESS_INFO:	We have output HDU locked
PROCESS_INFO:	We have output HDU setup
HERE
DEBUG: gpu = 0
DEBUG: npkt = 65536
DEBUG: nfft = 65536
DEBUG: nchan_fine = 4097
DEBUG: nthread = 128
DEBUG: nsamp_packed = 536870912
DEBUG: nsamp_fft = 268500992
PROCESS_INFO:	input buffer block size is 536870912 bytes, output buffer block size is 16388 bytes
PROCESS_INFO:	We have input buffer block size checked
PROCESS_INFO:	We have output buffer block size checked
PROCESS_INFO:	 device input buffer size is 8192d bytes
GPU free memory is 2940.1, total is 10016.8 MBbytes
We are at 0 block
Memory copy from host to device of 0 block done
We are at 1 block
Memory copy from host to device of 1 block done
pipeline   1916.450806 milliseconds, pipline with memory transfer averaged with 2 blocks
pipeline   98.315460 milliseconds, memory transfer h2d averaged with 2 blocks
pipeline   0.000000 milliseconds, memory transfer d2h averaged with 2 blocks
available  536.870912 milliseconds, available time for pipeline
========= ERROR SUMMARY: 0 errors
```

# Test the pipeline with real data

Real data is binary data with no packet header. As it has only one frequency channel and one polarisation, it is a pure time series with 8 bits signed integer and real sampling. Its bandwidth is 512 MHz, so the sampling rate is 1024 Mbps. All details can be found a dada header [file](header/512MHz_1ant1pol.header).

We can use `dada\_diskdb` to load binary data to ring buffer and let the pipeline process it. However, in order to go with this approach, we need to attach dada header at the beginning of binary data. As the binary data is 1048576000 bytes, we have 1.024 seconds real data. We first need to make sure that header file size is 4096 bytes and then cat it along with binary data to a new file. The commands are as follow (maybe there is a better to create a file with dada header ...):

```
truncate -s 4096 512MHz_1ant1pol.header
cat 512MHz_1ant1pol.header sat_samples_mini.bin > sat_samples_mini.dada
```

Now we can replace `dada\_jundb` with `dada\_diskdb` to feed real data to ring buffer.

```
dada_diskdb -k a000 -f sat_samples_mini.dada -s -z
```

We also need to replace `dada_dbnull` with `dada_dbdisk` to record pipeline output to files.

```
dada_dbdisk -k b000 -D . -z -o -s
```

Be careful here, `-z`and `-o` may not work sometimes.

Now we can check dada with python code, which will be discussed later. most of the result match Xiaoyun's result. 

## need to do
1. reboot machine to make CUDA works so that I can run the pipeline, done
2. need to get jupyter work so that I can run notebook to check Python result, do not need jupyter for now
3. check pipeline result with notebook result, working on that
4. I now have jupyter run there after I upgrade pip3, but
   1. it does not open broswer
   2. it asks password to move forward
5. check individual FFT output
6. check if there is an overflow

## possible cause of wrong result
1. header size is not 4096 bytes
2. scaling number is not the same as python numpy.fft.rfft, they are the same

## Found out so far
1. 32-bit has the same result as 64-bit in python
2. relative error in power comparing cuda and python is not significant
3. with sigma 64 as random input, still the same 
4. power data also looks reasonable

# something is wrong with power\_taccumulate kernel
its result does not match power\_taccumulate c code. maybe, but C and kernel code has small relative error, which is about 4E-7, why spec data has a very different result?

# overflaw of raw data?
# FFT different in CUDA and python?
1. checked so far do not see any problem
