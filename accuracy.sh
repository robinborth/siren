python evaluate.py --config=accuracy_abc.yaml 
python evaluate.py --config=accuracy_thingi10k.yaml 
python evaluate.py --config=accuracy_shapenet.yaml 
rsync -avz /home/borth/siren/eval/ thesis:/home/borth/neural-poisson/import/siren