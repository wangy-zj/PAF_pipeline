# udpgen

Run `udpgen` on odin and send data to the same network interface on odin is simple, just 
```
./udpgen -i 10.17.4.2 -p 12346 -I 10.17.4.2 -P 12345 -r 10
```

or with vma
```
LD_PRELOAD=libvma.so ./udpgen -i 10.17.4.2 -p 12346 -I 10.17.4.2 -P 12345 -r 10
```

# udp2db
Run `udp2db` needs to create a ring buffer first, assume that we want to hold 8192 packets at each ring buffer block, we need to run 
```
dada_db -k a000 -b 67108864 -p -w
```

or 65536 packets, we run 

```
dada_db -k a000 -b 536870912 -p -w
```

We also need to read data from that ring buffer, otherwise it blocks forever
```
dada_dbnull -k a000
```
or 
```
dada_dbdisk -k a000  -D . -z -o -s
```
to record data to disk. The second one is bad for bandwidth test, but good to check header contents.

Now we can run `udp2db`
```
./udp2db -i 10.17.4.2 -p 12345 -f /data/den15c/.udp-pipeline/header/512MHz_1ant1pol.header -k a000 -m 56400
```

# run pipeline with `udp2db`
We first need to create pipeline output ring buffer as we do [here](../README.md)
```
dada_db -k b000 -b 16388 -p -w 
```

then run to read pipeline output 
```
dada_dbnull -k b000
```

we then need to run `pipeline_dada_1ant1pol` instead of `dada_dbnull` as discussed [here](../README.md)
```
pipeline_dada_1ant1pol -i a000 -o b000 -g 0 -n 128
```

the pipeline runs smoothly with no memory issue or hang.

**there is a counter bug in `udp2db`???**

# Run pipeline with 4096+8 Bytes packets

- Run without network data stream
```
dada_db -k a000 -b 536870912 -p -w -l
dada_db -k b000 -b 16388 -p -w -l 
dada_dbnull -k b000 -z

./build/pipeline/pipeline_dada_1ant1pol -i a000 -o b000 -n 128 -g 0
dada_junkdb -k a000 -n header/512MHz_1ant1pol_4096B.header -r 64000 -t 1
```

- Run with network data stream
```
dada_db -k a000 -b 536870912 -p -w -l
dada_db -k b000 -b 16388 -p -w -l 
dada_dbnull -k b000 -z

./build/pipeline/pipeline_dada_1ant1pol -i a000 -o b000 -n 128 -g 0
LD_PRELOAD=libvma.so ./build/udp/udpgen -i 10.17.4.2 -p 12346 -I 10.17.4.2 -P 12345 -r 10
LD_PRELOAD=libvma.so ./build/udp/udp2db -i 10.17.4.2 -p 12345 -f /data/BALDR_0/den15c/.udp-pipeline/header/512MHz_1ant1pol_4096B.header -k a000 -m 56400
```
