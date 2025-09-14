#!/bin/bash

scale="scale=w=2560:h=-2:flags=lanczos"
fpsin=60
fpsout=60
frame_multiplier=4.0
codec="-c:v libx264 -preset veryslow -tune animation -crf 17 -pix_fmt yuv420p -movflags +faststart"
filter="setpts=$frame_multiplier*PTS,fps=$fpsout,$scale"
name=$(basename "$1") # Get the name of the directory passed as an argument, not the full path

# ffmpeg -y -r $fpsin -i "frames/frame_%04d.png" -filter_complex "fps=$fpsout,$scale" $codec "animation.mp4"
# ffmpeg -y -r $fpsin -i "$1/frame_%04d.png" -filter_complex "fps=$fpsout,$scale" $codec "animation.mp4"
ffmpeg -y -r $fpsin -i "$1/frame_%05d.png" -filter_complex "fps=$fpsout,$scale" $codec "animations/$name.mp4"
