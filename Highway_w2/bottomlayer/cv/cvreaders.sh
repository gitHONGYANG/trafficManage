python /home/highway/Highway/bottomlayer/core/get_tvs.py > /home/highway/Highway/bottomlayer/cv/log/get_tvs.txt 2>&1

: > /home/highway/Highway/bottomlayer/core/process.txt
cat /home/highway/Highway/bottomlayer/core/tvs.txt | while read line
do
    tv=${line}
    echo ${tv}
    nohup python /home/highway/Highway/bottomlayer/cv/cv_basereader.py ${tv}  > /home/highway/Highway/bottomlayer/cv/log/${tv}.txt 2>&1 &
    echo $! >> /home/highway/Highway/bottomlayer/core/process.txt
done

#nohup python /home/highway/Highway/bottomlayer/cv/cv_basereader.py 'TV57'  > /home/highway/Highway/bottomlayer/cv/log/log57.txt 2>&1 &
