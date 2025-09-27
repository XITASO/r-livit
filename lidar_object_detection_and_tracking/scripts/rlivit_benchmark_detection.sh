CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pointpillar.yaml --extra_tag experiment_00
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pointpillar.yaml --extra_tag experiment_01
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pointpillar.yaml --extra_tag experiment_02

CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pointrcnn.yaml --extra_tag experiment_00
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pointrcnn.yaml --extra_tag experiment_01
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pointrcnn.yaml --extra_tag experiment_02

CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pv_rcnn.yaml --extra_tag experiment_00
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pv_rcnn.yaml --extra_tag experiment_01
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/pv_rcnn.yaml --extra_tag experiment_02

CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/voxel_rcnn.yaml --extra_tag experiment_00
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/voxel_rcnn.yaml --extra_tag experiment_01
CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py --cfg_file ./tools/cfgs/r_livit_models/voxel_rcnn.yaml --extra_tag experiment_02

