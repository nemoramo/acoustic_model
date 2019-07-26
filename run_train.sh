horovodrun -np 8 -H localhost:8 python train.py \
        --data_type all --model_path logs --model_name redrop_am.pt \
        --gpu_rank 8 --epochs 1000 --save_step 20 --load_model --batch_size 16

horovodrun -np 8 -H localhost:8 python train_batchrnn.py \
        --data_type thchs --model_path logs --model_name batchrnn_am.pt \
        --gpu_rank 8 --epochs 1000 --save_step 20 --batch_size 16