This is the official implementation for the following paper:

[Protein Multimer Structure Prediction via Prompt Learning](https://arxiv.org/abs/2402.18813), *ICLR 2024*.


## Data Processing
First, download the latest Biological assembly PDB files from the RCSB PDB database. We recommend running the command:

```angular2html
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/biounit/PDB/all/ ./pdb_all
```

1. ```cd cdhit_process```

2. Run ```python pdb2fasta.py```. You can download the FASTA file for all PDBs from [here](https://drive.google.com/file/d/1AbpgTIU2tZen9O90ScfxAQMnmMoi9HGe/view?usp=drive_link) and place it in ```.cdhit_process/fasta_all.txt```.

3. Run ```cd-hit -i fasta_all.txt -o cluster_result -c 0.40 -n 2```. The clustering result is stored in ```cdhit_process/cluster_result.clstr```.

4. Run ```selec_from_clustering.py```. We have the rough version of the PDB-M dataset ```./PDB-M/PDB-M-rough.txt```.

5. Run ```python pre_process/process_source_data.py -n_min 3 -n_max 30 -homo_ratio 0.99 -data_fraction 1.0```

After this process, we obtain the PDB-M dataset ```./source_data/all_chain_pdb.txt```, which is also the ```./PDB-M/PDB-M.txt```. We randomly generate the training set ```./PDB-M/PDB-M-train.txt``` and the test set ```./PDB-M/PDB-M-test.txt```.

## Source Data
1. Creating the pre-training source data with multimers of $N=3, 4, 5$.
```angular2html
python ./source_data/process_source_data.py -n_min 3 -n_max 5 -homo_ratio 0.5 -data_fraction 1.0
```

2. Creating the dgl format data and labels for training.

```angular2html
python produce_training_dgls.py
```
After processing the source data, ```train_oracle_dgl_train_3_5.pt``` and ```rmsd_loss_train_3_5.pt``` will appear in ```./source_data```.

## Target Data

Creating the prompting target data with multimers of $N=3\sim30$ (training set).

```angular2html
python produce_prompting_dgls.py
```
After processing the target data, ```train_prompt_dgls.bin``` and ```train_prompt_rmsd.pt``` will appear in ```./target_data```.

This process is time consuming. If you want to omit it, you can download the files ```train_prompt_dgls.bin```, ```train_prompt_rmsd.pt``` and ```new_node_emb.pt``` directly from [here](https://drive.google.com/drive/folders/12kQvZrnfO90qYEaFWz8s5U12QYYcXpK9?usp=drive_link) and put them in the ```./target_data```.

## Preparing Dimers of GT and ESMFold

For getting GT dimers, we can handle the pair of chains without physical contact with EquiDock. 

```angular2html
./dimer/inference_rigid_half_euidock.py.py
```

However, this is a bit time consuming (because we need all possible pairs of chains within each multimer). As an alternative, we quickly generate dimers for pairs without physical contact, as long as the dimers of the two chains come into contact with each other.

```angular2html
./dimer/inference_rigid_no_euidock_fast.py
```

We need to prepare the ESMFold-produced dimers for test set multimers.

```angular2html
./dimer/inference_rigid_esmfold.py.py
```
## Pre-training

Training the GIN model
```angular2html
python run_pre_training.py -h_feats 512 -cls_h 256 -num_layers 1 -gnn_type 'gin' -lr 1e-3 -bs 50 -epochs 300
```

## Prompting
```angular2html
python run_prompting.py -h_feats 512 -lr 1e-3 -bs 3000 -epochs 50
```
## Inference (test)

Using ground-truth dimers:
```angular2html
python inference.py -dimer_type gt
```
Using ESMFold dimers:
```angular2html
python inference.py -dimer_type esmfold
```
