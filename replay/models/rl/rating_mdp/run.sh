# python3 mdp.py --top_k 10 --random_emb True --data_use 10000 --epochs 50 --pfunc bina --wandb_run_name random_10000_top10_p2_bina
# python3 mdp.py --top_k 10 --random_emb False --data_use 10000  --epochs 50 --pfunc bina --wandb_run_name als_10000_top10_p2_bina

# python3 mdp.py --top_k 10 --random_emb True --data_use 10000 --epochs 50 --pfunc nrba --wandb_run_name random_10000_top10_p2_nrba
# python3 mdp.py --top_k 10 --random_emb False --data_use 10000  --epochs 50 --pfunc nrba --wandb_run_name als_10000_top10_p2_nrba

# python3 mdp.py --top_k 10 --random_emb True --data_use 10000 --epochs 50 --pfunc nr --wandb_run_name random_10000_top10_p2_nr
# python3 mdp.py --top_k 10 --random_emb False --data_use 10000  --epochs 50 --pfunc nr --wandb_run_name als_10000_top10_p2_nr

python3 rating_experiment.py --top_k 10 --emb als --data_use 90000 --epochs 100 --pfunc o --wandb_run_name random_10000_top10_p2_o_repdat
#python3 mdp.py --top_k 10 --emb als --data_use 90000 --epochs 100 --pfunc o --wandb_run_name als_10000_top10_p2_o_repdat
#python3 mdp.py --top_k 10 --emb ddpg --data_use 90000 --epochs 100 --pfunc o --wandb_run_name ddpg_10000_top10_p2_o_repdat

#python3 mdp.py --top_k 10 --emb ddpg --data_use 50000 --epochs 100 --pfunc o --wandb_run_name ddpg_10000_top10_p2_o_repdat

#python3 mdp.py --top_k 10 --emb ddpg --data_use 10000 --epochs 100 --pfunc bina --wandb_run_name ddpg_10000_top10_p2_bina
#python3 mdp.py --top_k 10 --emb ddpg --data_use 10000 --epochs 100 --pfunc nrba --wandb_run_name ddpg_10000_top10_p2_nrba
#python3 mdp.py --top_k 10 --emb ddpg --data_use 10000 --epochs 100 --pfunc o --wandb_run_name ddpg_10000_top10_p2_o

#python3 mdp.py --top_k 10 --emb rand --data_use 10000  --epochs 100 --pfunc o --wandb_run_name ddpg_10000_top10_p2_o
#python3 mdp.py --top_k 10 --emb als --data_use 10000  --epochs 100 --pfunc o --wandb_run_name als_10000_top10_p2_o


#python3 mdp.py --top_k 10 --random_emb True --data_use 30000 --epochs 100 --pfunc o --wandb_run_name random_30000_top10_p2_o
#python3 mdp.py --top_k 10 --random_emb False --data_use 30000  --epochs 100 --pfunc o --wandb_run_name als_30000_top10_p2_o


