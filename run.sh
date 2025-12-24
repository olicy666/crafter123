python inference.py \
  --image_dir test/images/car.jpg \
  --ckpt_path ./checkpoints/model.ckpt \
  --ddim_steps 10 \
  --video_length 16 \
  --device 'cuda:0' \
  --height 320 --width 576 \
  --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --mode single_view_autotraj \