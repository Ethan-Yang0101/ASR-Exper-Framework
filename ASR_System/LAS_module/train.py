from LAS_Model.LAS_Listener import LAS_Listener
from LAS_Model.LAS_Attender import SingleHeadAttention
from LAS_Model.LAS_Speller import LAS_Speller
from LAS_Model.LAS_Model import LAS_Model
from torch.utils.data import DataLoader
from Joint_module.ModelTrainer import ModelTrainer
from Joint_module.WSJDataset import data_preprocessing
from Joint_module.ArgParser import get_args
from Joint_module.WSJDataset import WSJDataset
import warnings
import numpy as np
import torch
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def excute_model_training():
    # reproductivity
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    # load argument parser
    args = get_args()
    # load librispeech dataset
    train_dataset = WSJDataset.make_speech_dataset(args.char_filename,
                                                   args.train_data_scp_filename,
                                                   args.train_label_scp_filename,
                                                   args.apply_ctc_task)
    valid_dataset = WSJDataset.make_speech_dataset(args.char_filename,
                                                   args.valid_data_scp_filename,
                                                   args.valid_label_scp_filename,
                                                   args.apply_ctc_task)
    test_dataset = WSJDataset.make_speech_dataset(args.char_filename,
                                                  args.test_data_scp_filename,
                                                  args.test_label_scp_filename,
                                                  args.apply_ctc_task)
    # create dataloaders
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=data_preprocessing,
                              worker_init_fn=seed_worker,
                              generator=g,
                              **kwargs)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=data_preprocessing,
                              worker_init_fn=seed_worker,
                              generator=g,
                              **kwargs)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=data_preprocessing,
                             worker_init_fn=seed_worker,
                             generator=g,
                             **kwargs)
    dataloaders = {'train_loader': train_loader,
                   'valid_loader': valid_loader,
                   'test_loader': test_loader}
    # create a LAS model
    listener = LAS_Listener(args.listener_input_size,
                            args.listener_hidden_size,
                            args.listener_num_layers,
                            args.listener_vocab_size,
                            args.listener_dropout_rate,
                            args.listener_bidirectional,
                            args.listener_rnn_type)
    if args.attention_type == 'single':
        attender = SingleHeadAttention(args.attender_encoder_state_dim,
                                       args.attender_hidden_dim,
                                       args.attender_share_mapping)
    speller = LAS_Speller(args.speller_vocab_size,
                          args.speller_embedding_dim,
                          args.speller_hidden_size,
                          args.speller_num_layers,
                          args.speller_encoder_context_dim,
                          args.speller_rnn_type,
                          args.speller_apply_encoder_context,
                          args.label_smoothing_rate)
    LAS_model = LAS_Model(listener, attender, speller,
                          args.apply_ctc_task, args.lambda_factor)
    LAS_model = set_gpu_device(args, LAS_model)
    optimizer = torch.optim.Adam(
        LAS_model.parameters(), lr=args.learning_rate)
    # create a model trainer
    solver = ModelTrainer(dataloaders, LAS_model, optimizer, args)
    solver.train_val_test_model()
    return


def set_gpu_device(args, model):
    device = None
    if torch.cuda.device_count() > 1 and args.use_gpu:
        device = torch.cuda.current_device()
        model.to(device)
        model = torch.nn.DataParallel(module=model)
        print('Use Multi GPU', device, '\n')
    elif torch.cuda.device_count() == 1 and args.use_gpu:
        device = torch.cuda.current_device()
        model.to(device)
        print('Use GPU', device, '\n')
    else:
        device = torch.device('cpu')
        model.to(device)
        print('use CPU only\n')
    return model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    excute_model_training()
