CXX=g++ \
python scripts/inference_debug.py \
--exp_dir='./experiment/inference/debug_2_' \
--resize_factor=2 \
--n_outputs_to_generate=5 \
--checkpoint_path='./experiment/baye_batch2/checkpoints/best_model.pt' \
--data_path='/home/shwan/COWORK/CelebA-HQ/test-debug/' \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs
