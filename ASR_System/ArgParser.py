import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Model hyperparameter
    parser.add_argument('--listener_vocab_size', type=int, default=32)
    parser.add_argument('--speller_vocab_size', type=int, default=32)
    parser.add_argument('--listener_input_size', type=int, default=120)
    parser.add_argument('--rnn_hidden_size', type=int, default=320)
    # Model MLT parameter
    parser.add_argument('--apply_ctc_task', type=int, default=1)
    parser.add_argument('--lambda_factor', type=float, default=0.2)
    # attender hyperparameter
    parser.add_argument('--attention_type', type=str, default='location')
    parser.add_argument('--sharpening_factor', type=float, default=2.0)
    # heuristics
    parser.add_argument('--listener_dropout_rate', type=float, default=0.0)
    parser.add_argument('--label_smoothing_rate', type=float, default=0.0)
    parser.add_argument('--use_sampling', type=float, default=False)
    parser.add_argument('--sampling_rate', type=float, default=0.0)
    parser.add_argument('--apply_noise', type=bool, default=True)
    parser.add_argument('--noise_ratio', type=float, default=0.1)
    # data preparation
    parser.add_argument('--char_filename', type=str,
                        default='./data/WallStreet/char.txt')
    parser.add_argument('--train_data_scp_filename', type=str,
                        default='./data/WallStreet/si284-0.9-train.fbank.scp')
    parser.add_argument('--train_label_scp_filename', type=str,
                        default='./data/WallStreet/si284-0.9-train.bchar.scp')
    parser.add_argument('--valid_data_scp_filename', type=str,
                        default='./data/WallStreet/si284-0.9-dev.fbank.scp')
    parser.add_argument('--valid_label_scp_filename', type=str,
                        default='./data/WallStreet/si284-0.9-dev.bchar.scp')
    parser.add_argument('--test_data_scp_filename', type=str,
                        default='./data/WallStreet/si284-0.9-dev.fbank.scp')
    parser.add_argument('--test_label_scp_filename', type=str,
                        default='./data/WallStreet/si284-0.9-dev.bchar.scp')
    # training process
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--save_checkpoint', type=bool, default=True)
    parser.add_argument('--save_folder_path', type=str, default='./model')
    parser.add_argument('--checkpoint_model_path', type=str,
                        default='./model/0.0_0.0_0.0_epoch14.pth.tar')
    # recognize process
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--best_hypo_num', type=int, default=1)
    parser.add_argument('--decode_max_len', type=int, default=0)
    args = parser.parse_args()
    return args
