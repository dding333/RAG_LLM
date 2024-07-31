#!/bin/bash

# 

URL=registry.cn-shanghai.aliyuncs.com

IMAGE_NAME=monkey

VERSION=0.1

docker build -t $URL/$IMAGE_NAME:$VERSION .




