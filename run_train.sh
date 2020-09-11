CUDA_VISIBLE_DEVICES=4 python run.py                  \
--train_data_path /home/ubuntu/dialoglue/banking/train_10.csv          \
--mlm_data_path /home/ubuntu/dialoglue/banking/train_10.csv     \
           --val_data_path /home/ubuntu/dialoglue/banking/val.csv   \
                         --test_data_path /home/ubuntu/dialoglue/banking/test.csv \
                                --token_vocab_path /home/ubuntu/checkpoints/bert_reddit_4_24/vocab.txt \
                                # num_epochs = 0 -> run eval \
                   --train_batch_size 16 --grad_accum 4 --dropout 0.1 --num_epochs 50 --learning_rate 6e-5 \
                                    --model_name_or_path /home/ubuntu/checkpoints/bert_reddit_4_24 --task intent --do_lowercase --max_seq_length 120 \
                               --repeat 5 --patience 10 --mlm_pre --mlm_during --use_observers --example

