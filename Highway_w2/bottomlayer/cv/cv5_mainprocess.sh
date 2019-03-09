export CUDA_VISIBLE_DEVICES=1
nohup python /home/highway/Highway/bottomlayer/cv5_mainprocess.py > /home/highway/Highway/bottomlayer/cv/log/cv_mainprogrocess5.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
