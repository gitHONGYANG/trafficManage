export PATH=/home/highway/anaconda3/bin:$PATH

nohup python /home/highway/HighWay/toplayer/tv_switch.py > /home/highway/HighWay/toplayer/log/tv_switch.txt 2>&1 &

nohup python /home/highway/HighWay/toplayer/client_service.py > /home/highway/HighWay/toplayer/log/client_service.txt 2>&1 &
nohup python /home/highway/HighWay/toplayer/service_w2.py > /home/highway/HighWay/toplayer/log/service_w2.txt 2>&1 &
nohup python /home/highway/HighWay/toplayer/service_w3.py > /home/highway/HighWay/toplayer/log/service_w3.txt 2>&1 &
nohup python /home/highway/HighWay/toplayer/service_w4.py > /home/highway/HighWay/toplayer/log/service_w4.txt 2>&1 &
nohup python /home/highway/HighWay/toplayer/service_w5.py > /home/highway/HighWay/toplayer/log/service_w5.txt 2>&1 &
