export PATH=/home/highway/anaconda3/bin:$PATH

export CUDA_VISIBLE_DEVICES=3
sleep 8

nohup python /home/highway/Highway/bottomlayer/rtsp_mainprocess.py 3 > /home/highway/Highway/bottomlayer/rtsp/log/rtsp_mainprogrocess4.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
