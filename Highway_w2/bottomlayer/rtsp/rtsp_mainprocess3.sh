export PATH=/home/highway/anaconda3/bin:$PATH

export CUDA_VISIBLE_DEVICES=2
sleep 6
nohup python /home/highway/Highway/bottomlayer/rtsp_mainprocess.py 2 > /home/highway/Highway/bottomlayer/rtsp/log/rtsp_mainprogrocess3.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
