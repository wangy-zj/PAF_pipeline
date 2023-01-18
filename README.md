# udp-pipeline

## 目录说明

header：dada格式的header文件

pipeline：GPU中运行的程序

udp：将udp包合并为dada格式

src和include：依赖的C文件和头文件

## 修改scripts目录下的pipeline.sh脚本

       1. ringbuffer
       
       raw（原始输入），bmf（波束合成之后），zoo（zoom fft之后）
       
       1）设置ringbuffer关键字： a000， b000, c000
       
       2）设置程序运行的线程数
       
       3）设置ringbffer大小： bufsz_raw, bufsz_bmf, bufsz_zoo
       
       计算各ringbuffer的大小
       
       输入（a000）：npkt×nelement×npol×pkt_data×sizeof(int8_t)
       
       beamform输出功率（b000）： npkt×pkt_nsamp×beam×bf_nchan×sizeof(float)/naverage_bf
       
       zoomfft输出功率（c000）：zoom_nsamp×zoom_nchan×bf_nchan×sizeof（float）/naverage_zoom

       
       4）根据设置参数创建3个ringbuffer
        
       2. 数据存储
       
       设置数据存储目录：dir_raw, dir_bmf, dir_zoo
       
       激活dada_dbdisk存储数据：dada_dbdisk -D 'save_dir' -k 'ringbuffer' -W
       
       3. 开启 pipeline 和 udp2db
       ../build/pipeline/pipeline_dada_beamform -i $key_raw -o $key_bmf -n $nreader_raw -g 0
       ../build/udp/udp2db -k $key_raw -i 10.11.4.54 -p 12345 -f ../header/512MHz_1ant1pol_4096B.header -m 56400
       
       4. 接收结束后关闭数据存储，清除ringbuffer
       
## 运行
       bash pipeline.sh
       udpgen -i 10.11.4.54 -p 12346 -I 10.11.4.54 -P 12345 -r 10   （ip和port根据实际情况修改，-i和-p为发送端，-I和-P为接收端）

