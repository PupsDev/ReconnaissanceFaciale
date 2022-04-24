#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="ressource/dataset/database25/*"
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image $f
  # cat "$f"
done