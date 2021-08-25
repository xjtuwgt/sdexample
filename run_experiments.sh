#!/usr/bin/env bash
#HOME=/mnt/cephfs2/nlp/guangtao.wang
#CONDA_ROOT=${HOME}/anaconda3
#PYTHON_VIRTUAL_ENVIRONMENT=hotpotqa
#source ${CONDA_ROOT}/etc/profile.d/conda.sh
#conda activate $PYTHON_VIRTUAL_ENVIRONMENT

eval "$(conda shell.bash hook)"
#conda activate hotpotqa

JOBS_PATH=toy_example_jobs
LOGS_PATH=toy_example_logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g2.q -l gpu=1 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 20
done