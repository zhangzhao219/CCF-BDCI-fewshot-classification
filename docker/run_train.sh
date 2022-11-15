directory=`pwd`

echo $directory

chmod -R 777 $directory

docker import ${directory}/image/Final.tar pain_275749:v1

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_10_22_19_12_04-3-0.62876738448.sh'

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_10_27_07_38_29_3-0.63077875449-8.sh'

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_11_01_04_26_32-3-0.63293263685.sh'

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_11_03_19_41_25-a-0.63234589689.sh'

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_11_05_05_55_17-0-0.62600679310.sh'

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_11_06_04_35_15-a-0.62673116125.sh'

docker run -v ${directory}/data:/mnt --gpus all --shm-size=6g -it pain_275749:v1 /bin/bash -c 'cd /mnt/code/train && bash 2022_11_06_19_08_24-a-.sh'