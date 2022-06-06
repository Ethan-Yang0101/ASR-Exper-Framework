from CTC_Attent.Listener import Listener
from CTC_Attent.Attender import LocationAwareAttention
from CTC_Attent.Speller import Speller
from CTC_Attent.Model import Model
from torch.utils.data import DataLoader
from ModelTrainer import ModelTrainer
from WSJDataset import data_preprocessing
from ArgParser import get_args
from WSJDataset import WSJDataset
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
    # load dataset
    train_dataset = WSJDataset.make_speech_dataset(
        char_filename=args.char_filename,
        data_scp_filename=args.train_data_scp_filename,
        label_scp_filename=args.train_label_scp_filename,
        apply_ctc_task=args.apply_ctc_task)
    valid_dataset = WSJDataset.make_speech_dataset(
        char_filename=args.char_filename,
        data_scp_filename=args.valid_data_scp_filename,
        label_scp_filename=args.valid_label_scp_filename,
        apply_ctc_task=args.apply_ctc_task)
    test_dataset = WSJDataset.make_speech_dataset(
        char_filename=args.char_filename,
        data_scp_filename=args.test_data_scp_filename,
        label_scp_filename=args.test_label_scp_filename,
        apply_ctc_task=args.apply_ctc_task)
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
    # create a listener
    listener = Listener(input_size=args.listener_input_size,
                        hidden_size=args.rnn_hidden_size,
                        vocab_size=args.listener_vocab_size,
                        dropout_rate=args.listener_dropout_rate)
    # create a attender
    if args.attention_type == 'location':
        attention = LocationAwareAttention
    attender = attention(d_model=args.rnn_hidden_size,
                         attention_type=args.attention_type,
                         sharpening_factor=args.sharpening_factor)
    # create a speller
    speller = Speller(vocab_size=args.speller_vocab_size,
                      embedding_dim=args.rnn_hidden_size,
                      hidden_size=args.rnn_hidden_size,
                      label_smoothing_rate=args.label_smoothing_rate)
    # create a model
    CTC_Attent_model = Model(listener=listener,
                             attender=attender,
                             speller=speller,
                             apply_ctc_task=args.apply_ctc_task,
                             lambda_factor=args.lambda_factor)
    CTC_Attent_model = set_gpu_device(args, CTC_Attent_model)
    # create an optimizer
    if args.optimizer == 'adam':
        optim = torch.optim.Adam
    optimizer = optim(CTC_Attent_model.parameters(), lr=args.learning_rate)
    # create a model trainer
    solver = ModelTrainer(dataloaders=dataloaders,
                          model=CTC_Attent_model,
                          optimizer=optimizer,
                          args=args)
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
