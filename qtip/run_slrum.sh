docker run -d -it --name qtip \
  -v /home/minkyu4506/:/home/minkyu4506/ \
  -v /data/:/data/ \
  --gpus all \
  --shm-size=400G \
  jegwangryu/qtip

docker exec qtip bash -c "cd /home/mkkim/Weight_compression/qtip && bash run_3_8b.sh"

docker stop qtip
