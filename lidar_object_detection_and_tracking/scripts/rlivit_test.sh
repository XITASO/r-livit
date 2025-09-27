# CUDA_VISIBLE_DEVICES=2 python3 ./tools/test.py --cfg_file ./tools/cfgs/r_livit_models/pointpillar.yaml --ckpt /home/user/workspace/checkpoints/pointpillar_7728.pth --batch_size 1 
# CUDA_VISIBLE_DEVICES=2 python3 ./tools/test.py --cfg_file ./tools/cfgs/kitti_models/pointpillar.yaml --ckpt /home/user/workspace/checkpoints/pointpillar_7728.pth --batch_size 1

CUDA_VISIBLE_DEVICES=2 python3 ./tools/test.py --cfg_file ./tools/cfgs/r_livit_models/pointpillar.yaml --ckpt /home/user/workspace/output/tools/cfgs/r_livit_models/pointpillar/experiment_00/ckpt/latest_model.pth --batch_size 1 