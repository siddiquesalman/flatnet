CHECKPOINT='flatnet'
MODEL_ROOT='flatnet'
INIT='Transpose'
python main.py --wtmse 1 --wtp 1.2 --wta 0.2 --disPreEpochs 0 --numEpoch 20 --modelRoot $MODEL_ROOT --init $INIT
