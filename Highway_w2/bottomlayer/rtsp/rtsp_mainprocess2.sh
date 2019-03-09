export PATH=/home/highway/anaconda3/bin:$PATH

export CUDA_VISIBLE_DEVICES=1
sleep 4
nohup python /home/highway/Highway/bottomlayer/rtsp_mainprocess.py 1 > /home/highway/Highway/bottomlayer/rtsp/log/rtsp_mainprogrocess2.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
