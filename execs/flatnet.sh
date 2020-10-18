CHECKPOINT='flatnet'
MODEL_ROOT='flatnet'
INIT='Transpose'
python main.py --wtmse 1 --wtp 0 --wta 0.6 --disPreEpochs 5 --numEpoch 25 --modelRoot $MODEL_ROOT --init $INIT
