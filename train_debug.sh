python scripts/train.py \
--dataset_type=celebs_super_resolution \
--exp_dir=experiment/debug \
--batch_size=2 \
--workers=2 \
--test_batch_size=2 \
--test_workers=2 \
--val_interval=1000 \
--save_interval=2000 \
--encoder_type=BayesianGradualStyleEncoder \
--mc_samples=5 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--w_norm_lambda=0.005 \
--resize_factors=1,2,4,8,16,32
