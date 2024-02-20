
echo DrugCell SETTINGS
echo SETTINGS
PROCS=6
export PROCS=6
# General Settings
export PPN=6
export WALLTIME=48:00:00
export NUM_ITERATIONS=1
export POPULATION_SIZE=75
# GA Settings
export STRATEGY='mu_plus_lambda'
export OFF_PROP=0.5
export MUT_PROB=0.8
export CX_PROB=0.2
export MUT_INDPB=0.5
export CX_INDPB=0.5
export TOURNAMENT_SIZE=4
# Lambda Settings
export CANDLE_CUDA_OFFSET=4
export CANDLE_DATA_DIR=/tmp/DrugCell/Data
# Polaris Settings
# export QUEUE="debug"

