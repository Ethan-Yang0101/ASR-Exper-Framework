python evaluate.py  --apply_ctc_task True \
                    --lambda_factor 0.2 \
                    --listener_vocab_size 34 \
                    --listener_input_size 120 \
                    --listener_hidden_size 256 \
                    --listener_num_layers 1 \
                    --listener_bidirectional True \
                    --listener_rnn_type 'lstm' \
                    --attention_type 'single' \
                    --attender_encoder_state_dim 512 \
                    --attender_hidden_dim 512 \
                    --attender_share_mapping True \
                    --speller_vocab_size 33 \
                    --speller_embedding_dim 512 \
                    --speller_hidden_size 512 \
                    --speller_num_layers 2 \
                    --speller_encoder_context_dim 512 \
                    --speller_rnn_type 'lstm' \
                    --speller_apply_encoder_context False \
                    --char_filename './data/WallStreet/char.txt' \
                    --train_data_scp_filename './data/WallStreet/si284-0.9-train.fbank.scp' \
                    --train_label_scp_filename './data/WallStreet/si284-0.9-train.bchar.scp' \
                    --valid_data_scp_filename './data/WallStreet/si284-0.9-dev.fbank.scp' \
                    --valid_label_scp_filename './data/WallStreet/si284-0.9-dev.bchar.scp' \
                    --test_data_scp_filename './data/WallStreet/si284-0.9-dev.fbank.scp' \
                    --test_label_scp_filename './data/WallStreet/si284-0.9-dev.bchar.scp' \
                    --use_gpu True \
                    --optimizer 'adadelta' \
                    --learning_rate 1.0 \
                    --batch_size 16 \
                    --num_epochs 20 \
                    --maxnorm 5.0 \
                    --save_checkpoint True \
                    --save_folder_path './baseline' \
                    --checkpoint_model_path './baseline/epoch19.pth.tar' \
                    --beam_size 1 \
                    --best_hypo_num 1 \
                    --decode_max_len 150 \
                    --listener_dropout_rate 0.0 \
                    --label_smoothing_rate 0.0