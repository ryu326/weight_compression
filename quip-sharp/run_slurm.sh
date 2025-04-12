docker run -d -it --name quip \
  -v /home/minkyu4506/:/home/minkyu4506/ \
  -v /data/:/data/ \
  --gpus all \
  --shm-size=400G \
  jegwangryu/quip

docker exec quip bash -c "cd (Weight_compression 디렉토리 경로)/quip-sharp && bash run3_8b.sh"

docker stop quip
