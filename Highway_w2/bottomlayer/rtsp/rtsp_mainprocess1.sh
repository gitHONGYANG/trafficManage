export PATH=/home/highway/anaconda3/bin:$PATH

export CUDA_VISIBLE_DEVICES=0
sleep 2

nohup python /home/highway/Highway/bottomlayer/rtsp_mainprocess.py 0 > /home/highway/Highway/bottomlayer/rtsp/log/rtsp_mainprogrocess1.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
