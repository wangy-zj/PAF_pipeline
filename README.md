# udp-pipeline

header：dada格式的header文件
pipeline：GPU中运行的程序
udp：将udp包合并为dada格式
src和include：依赖的C文件和头文件

## performance test

# 1. 修改scripts目录下的pipeline.sh脚本
       ringbuffer
       包含三个ringbuffer：raw（原始输入），bmf（波束合成之后），zoo（zoom fft之后）
       设置ringbuffer关键字： a000， b000, c000
       设置程序运行的线程数
       设置ringbffer大小： bufsz_raw, bufsz_bmf, bufsz_zoo
        根据设置参数创建3个ringbuffer
        
        数据存储
        设置数据存储目录：dir_raw, dir_bmf, dir_zoo
        设置

