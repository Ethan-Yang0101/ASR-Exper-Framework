from LAS_Model.LAS_Model import LAS_Model
from LAS_Model.LAS_BeamCoder import LAS_BeamCoder
from torch.utils.data import DataLoader
from Joint_module.WSJDataset import data_preprocessing
from Joint_module.ArgParser import get_args
from Joint_module.WSJDataset import WSJDataset
import matplotlib.pyplot as plt
import Levenshtein
import numpy as np
from tqdm import tqdm
import torch
import warnings


def evaluate_recognition_results():
    # load argument parser
    args = get_args()
    # load librispeech dataset
    test_dataset = WSJDataset.make_speech_dataset(
        args.char_filename,
        args.test_data_scp_filename,
        args.test_label_scp_filename,
        args.apply_ctc_task)
    # create dataloader
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=data_preprocessing,
                             **kwargs)
    # create a LAS model
    LAS_model = LAS_Model.load_model(args.checkpoint_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LAS_model.to(device)
    LAS_model.eval()
    listener = LAS_model.listener
    speller = LAS_model.speller
    attender = LAS_model.attender
    # create a beam search decoder
    char_dict = test_dataset.char_dict
    index_dict = {v: k for k, v in char_dict.items()}
    index_dict[char_dict['<space>']] = ' '
    index_dict[char_dict['<noise>']] = '*'
    index_dict[char_dict['<eos>']] = '$'
    index_dict[char_dict['<sos>']] = '#'
    LAS_beamcoder = LAS_BeamCoder(listener, attender, speller, None,
                                  index_dict, char_dict['<sos>'],
                                  char_dict['<eos>'], args)
    # compute word error rate and character error rate
    cer, wer = 0.0, 0.0
    running_cer, running_wer = 0.0, 0.0
    num_data = len(test_loader)
    pbar = tqdm(total=len(test_loader))
    for batch_index, batch_dict in enumerate(test_loader):
        # decode an utterance
        input_utterance = batch_dict['spectrograms'].to(device)
        target_label = batch_dict['target_labels'][0][:-1]
        ground_truth = ''.join([index_dict[int(x)] for x in target_label])
        input_length = batch_dict['input_lengths']
        nbest_hyps = LAS_beamcoder.beam_search_decoding(
            input_utterance, input_length)
        hyp = nbest_hyps[0]
        prediction = ''.join([index_dict[int(x)] for x in hyp['seq'][1:-1]])
        cer_t = compute_character_error_rate(ground_truth, prediction)
        wer_t = compute_word_error_rate(ground_truth, prediction)
        running_cer += (cer_t - running_cer) / (batch_index + 1)
        running_wer += (wer_t - running_wer) / (batch_index + 1)
        pbar.update(1)
        pbar.set_postfix_str(" cer: {:.4f}, wer: {:.4f}".format(
            running_cer, running_wer))
    message = "character error rate: {:.4f}\nword error rate: {:.4f}".format(
        running_cer, running_wer)
    pbar.write(message)
    return


def compute_character_error_rate(ground_truth, prediction):
    error = Levenshtein.distance(ground_truth, prediction)
    return error / len(ground_truth)


def compute_word_error_rate(ground_truth, prediction):
    ground_truth = ground_truth.split(' ')
    prediction = prediction.split(' ')
    error = Levenshtein.distance(ground_truth, prediction)
    return error / len(ground_truth)


def plot_learning_curve(model_path):
    package = torch.load(model_path)
    train_state = package['train_state']
    plt.plot(train_state['epoch_index'],
             train_state['train_att_loss'], label='train_att_loss')
    plt.plot(train_state['epoch_index'],
             train_state['train_ctc_loss'], label='train_ctc_loss')
    plt.plot(train_state['epoch_index'],
             train_state['train_loss'], label='train_loss')
    plt.plot(train_state['epoch_index'],
             train_state['val_loss'], label='val_loss')
    plt.xticks(np.arange(0, 18))
    plt.yticks(np.linspace(0, 2.3, 10))
    plt.xlabel('epoch number', fontsize=14)
    plt.ylabel('loss', fontsize=15)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    evaluate_recognition_results()
