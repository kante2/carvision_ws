# ---------호스트-------------------

cd ~/carvision_ws/docker_carvision

# 이미지 빌드
docker compose build

# 컨테이너 실행 (백그라운드)
docker compose up -d

# 또는 바로 bash로 붙기
docker compose run --rm carvision-noetic
# or
docker exec -it carvision-noetic bash



# ---------컨테이너-------------------------
cd /root/carvision_ws
ls   # README.md, docker_carvision, src 다 보여야 정상

# catkin 처음 빌드할 때
cd /root/carvision_ws
catkin_make
