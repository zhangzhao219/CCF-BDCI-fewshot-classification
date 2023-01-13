directory=`pwd`

echo $directory

chmod -R 777 $directory

docker import ${directory}/image/Final.tar pain_275749:v1

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/inference && bash inference_submit_after_train.sh'