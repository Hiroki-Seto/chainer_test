find ./conf -name "*.json" | xargs -P 3 -I {} sh -c 'MPLBACKEND=Agg python train.py -c {}'
