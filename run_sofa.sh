sudo docker build -t sofa:ubuntu-22.04 .

sudo docker run --rm -it \
  -v /opt/emio-labs/resources/sofa:/opt/sofa \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri \
  sofa:latest

