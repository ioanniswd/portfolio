#!/bin/bash

scale="scale=w=960:h=-1:flags=lanczos"  # Adjusted to smaller width for GIF
fpsin=60
fpsout=30                               # Lower FPS helps reduce file size
frame_multiplier=4.0                    # Controls speed
filter="setpts=${frame_multiplier}*PTS,fps=${fpsout},${scale}"
name=$(basename "$1")                   # Extract folder name
outdir="animations"
mkdir -p "$outdir"

# Temporary palette file
palette="/tmp/palette.png"

# Step 1: Generate palette
ffmpeg -y -r $fpsin -i "$1/frame_%05d.png" -vf "$filter,palettegen" "$palette"

# Step 2: Generate GIF using palette
ffmpeg -y -r $fpsin -i "$1/frame_%05d.png" -i "$palette" -lavfi "$filter [x]; [x][1:v] paletteuse" "$outdir/$name.gif"
