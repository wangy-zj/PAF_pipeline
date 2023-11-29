# udp-pipeline

## 目录说明

       header：dada格式的header文件，后续考虑参数写在.h文件中
       
       pipeline：波束合成+积分
       
       udp：接收发送udp包
       
       src和include：依赖的C文件和头文件

## 修改scripts目录下的pipeline.sh脚本

       1. ringbuffer
       
       raw（原始输入），bmf（波束合成之后）
       
       1）设置ringbuffer关键字： a000， b000
       
       2）设置ringbffer大小： bufsz_raw, bufsz_bmf
       
       计算各ringbuffer的大小
       
       输入（a000）：N_PKT_PER_BLOCK×N_ANTENNA×PKT_DTSZ×SAMPSZ
       
       beamform输出功率（b000）： N_TIMESTEP_PER_BLOCK×N_BEAM×N_CHAN×sizeof(float)/N_AVERAGE

       
       3）根据设置参数创建2个ringbuffer
        
       2. 数据存储
       
       设置数据存储目录：dir_raw, dir_bmf
       
       激活dada_dbdisk存储数据：dada_dbdisk -D 'save_dir' -k 'ringbuffer' -W
       
       3. 开启 pipeline 和 udp2db
       bash pipeline.sh
       
       4. 接收结束后关闭数据存储，清除ringbuffer
       
## 自发自收仿真
       bash pipeline.sh
       udpgen -n 10   （-n 参数为发送数据包的时间）

## 联系人
       王钰         ywang@zhejianglab.com
