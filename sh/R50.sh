#!/bin/bash


python CLIP_VIT_LONGTAIL.py \
  --arch clip_VIT \
  --mark clip_VIT_bt256 \
  -dataset herbariumDataset \
  --data_path /media/intisar/dataset1/visual_categorization/herbarium-2022-fgvc9/ \
  -b 256 \
  --epochs 200 \
  --num_works 20 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --beta 0.85 \
  --gamma 0.3 \
  --after_1x1conv \
  --num_classes 15505 
  
