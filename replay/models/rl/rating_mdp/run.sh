
python3 rating_experiment.py --top_k 10 --emb als --data_use 90000 --epochs 100 --pfunc nrs --wandb_run_name als_10000_top10_nrs_repdat &

python3 rating_experiment.py --top_k 10 --emb als --data_use 90000 --epochs 100 --pfunc o --wandb_run_name als_10000_top10_o_repdat 

python3 rating_experiment.py --top_k 10 --emb als --data_use 90000 --epochs 100 --pfunc nr --wandb_run_name als_10000_top10_nr_repdat &

python3 rating_experiment.py --top_k 10 --emb als --data_use 90000 --epochs 100 --pfunc mr --wandb_run_name als_10000_top10_mono_repdat 




python3 rating_experiment.py --top_k 10 --emb ddpg --data_use 90000 --epochs 100 --pfunc nrs --wandb_run_name ddpg_10000_top10_nrs_repdat &

python3 rating_experiment.py --top_k 10 --emb ddpg --data_use 90000 --epochs 100 --pfunc o --wandb_run_name ddpg_10000_top10_o_repdat 

python3 rating_experiment.py --top_k 10 --emb ddpg --data_use 90000 --epochs 100 --pfunc nr --wandb_run_name ddpg_10000_top10_nr_repdat &

python3 rating_experiment.py --top_k 10 --emb ddpg --data_use 90000 --epochs 100 --pfunc mr --wandb_run_name ddpg_10000_top10_mono_repdat 
