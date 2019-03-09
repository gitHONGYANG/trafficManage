export CUDA_VISIBLE_DEVICES=0
nohup python /home/highway/Highway/bottomlayer/cv1_mainprocess.py > /home/highway/Highway/bottomlayer/cv/log/cv1_mainprogrocess.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
