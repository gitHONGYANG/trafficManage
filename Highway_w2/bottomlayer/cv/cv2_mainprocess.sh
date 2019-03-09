export CUDA_VISIBLE_DEVICES=1
nohup python /home/highway/Highway/bottomlayer/cv2_mainprocess.py > /home/highway/Highway/bottomlayer/cv/log/cv_mainprogrocess2.txt 2>&1 &
echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
