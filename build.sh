#!/bin/sh

time_tag=`date +"%Y-%m-%d-%H-%M"`
git_tag=`git rev-parse --short HEAD`
image_tag="${time_tag}-${git_tag}"
# image_name=registry-vpc.cn-shanghai.aliyuncs.com/maim1/whisper-rs
# image_name=registry.cn-shanghai.aliyuncs.com/maim1/whisper-rs
image_name=whisper-rs
docker build -t $image_name:$image_tag .
docker tag $image_name:$image_tag $image_name:latest
# docker push $image_name:$image_tag
# docker push $image_name:latest
echo $image_tag

