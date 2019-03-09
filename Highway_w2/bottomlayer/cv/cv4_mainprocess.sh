export CUDA_VISIBLE_DEVICES=3
nohup python /home/highway/Highway/bottomlayer/cv4_mainprocess.py > /home/highway/Highway/bottomlayer/cv/log/cv_mainprogrocess4.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
