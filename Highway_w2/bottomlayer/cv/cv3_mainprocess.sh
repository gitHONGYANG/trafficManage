export CUDA_VISIBLE_DEVICES=2
nohup python /home/highway/Highway/bottomlayer/cv3_mainprocess.py > /home/highway/Highway/bottomlayer/cv/log/cv_mainprogrocess3.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
