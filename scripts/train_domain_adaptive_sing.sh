#!/bin/bash

singularity exec --nv \
    --bind /data/users/babdullah/projects/speech_cls:/speech_cls \
    --bind /data/users/babdullah/data:/data \
    /data/users/babdullah/projects/SLR-GP/lang_id.sif bash /speech_cls/scripts/train_baseline_LID_mfsc_sbc.sh \
    2> /data/users/babdullah/projects/speech_cls/logs/${JOB_ID}.err.log \
    1> /data/users/babdullah/projects/speech_cls/logs/${JOB_ID}.out.log
