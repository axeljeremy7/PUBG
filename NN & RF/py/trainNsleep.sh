#!/bin/bash

source activate stat

echo -e "\n"

python NN_small.py

id=$!
wait $id

python NN_small_FE.py

id=$!
wait $id

python NN_full.py

id=$!
wait $id

python NN_full_FE.py

id=$!
wait $id

source deactivate stat