# Attribution of Gradient Based Adversarial Attacks for Reverse Engineering of Deceptions

Code to replicate results in this paper: https://arxiv.org/pdf/2103.11002.pdf

1. Create a directory called "data/" in the root of this repo
2. Download the dataset [here](https://bisque.ece.ucsb.edu/client_service/view?resource=https://bisque.ece.ucsb.edu/data_service/00-mBPHmnPiWe6wuicnAvqJDU)
3. Unzip into the "data/" directory
4. Create a virtualenv, and install the packages in requirements.txt
5. Create the adversarial samples by executing run_attacks.py, in src/data/
6. Train each of the 3 models in the paper, by running the following in src/models/

    python train.py co_occur GPU_ID

    python train.py laplace GPU_ID

    python train.py direct GPU_ID

7. Produce test results by running this for all 3 methods:

    python predict.py {co_occur,laplace,direct} GPU_ID
    
8. Print out results and confusion matrices as arrays and latex code:

    python eval.py {co_occur,laplace,direct}

