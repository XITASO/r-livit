## Modification part
NAME="r_livit"
DETECTOR="pointpillar"
DATA_SOURCE_PATH="/mnt/data/kitti/r_livit_tracking_split/testing/"
TRACK_EVAL_GT_PATH="${DATA_SOURCE_PATH}/label"
TRACK_EVAL_CALIB_PATH="${DATA_SOURCE_PATH}/calib"

EXPERIMENT_NAME="exp1"
DET_OUTPUT="output/detection/${DETECTOR}"  # folder of detection results containing json-format detection results
OUTPUT_PATH="output/tracking/${NAME}_simpletrack_${EXPERIMENT_NAME}"  # folder of tracking results


## Run part

OUTPUT_PATH_DTC="output/detection/${DETECTOR}_converted"
OUTPUT_PATH_TRACK="${OUTPUT_PATH}/tracking_results_to_kitti"
TRACK_EVAL_OUTPUT_PATH="${OUTPUT_PATH}/tracking_evaluation_results"


# ### Convert detection results to KITTI format
# echo """Convert detection results to KITTI format"""
# mkdir -p $OUTPUT_PATH_DTC
# python3 adapters/ab3dmot_adapter.py convert \
#   --input-dir-path $DET_OUTPUT \
#   --output-dir-path $OUTPUT_PATH_DTC \
#   --ori-path $DATA_SOURCE_PATH \

CAT="Car"
### Run Tracking
# echo """Run tracking"""
mkdir -p "${OUTPUT_PATH_TRACK}_${CAT}"
python3 adapters/mot3d_adapter.py \
  --obj_type $CAT \
  --data_folder $DATA_SOURCE_PATH \
  --gt_folder $TRACK_EVAL_GT_PATH \
  --det_data_folder $OUTPUT_PATH_DTC  \
  --result_folder "${OUTPUT_PATH_TRACK}_${CAT}"
 

### Evaluate Tracking
echo """Evaluate tracking results"""
mkdir -p "${TRACK_EVAL_OUTPUT_PATH}_${CAT}"
python3 adapters/ab3dmot_adapter.py eval \
  --track_eval_gt_path $TRACK_EVAL_GT_PATH \
  --calib_gt_path $TRACK_EVAL_CALIB_PATH \
  --track_results_path "${OUTPUT_PATH_TRACK}_${CAT}" \
  --track_eval_output_path "${TRACK_EVAL_OUTPUT_PATH}_${CAT}" \
  --cat $CAT
