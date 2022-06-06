from CTC_Attent.Model import Model
from CTC_Attent.BeamCoder import BeamCoder
from torch.utils.data import DataLoader
from WSJDataset import data_preprocessing
from ArgParser import get_args
from WSJDataset import WSJDataset
import matplotlib.pyplot as plt
import torch
import warnings


def get_recognition_results(sample_indices):
    # load argument parser
    args = get_args()
    # load test dataset
    test_dataset = WSJDataset.make_speech_dataset(
        args.char_filename,
        args.test_data_scp_filename,
        args.test_label_scp_filename,
        apply_ctc_task=args.apply_ctc_task)
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
    CTC_Attent_model = Model.load_model(args.checkpoint_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CTC_Attent_model.to(device)
    CTC_Attent_model.eval()
    listener = CTC_Attent_model.listener
    speller = CTC_Attent_model.speller
    attender = CTC_Attent_model.attender
    # create a beam search decoder
    char_dict = test_dataset.char_dict
    index_dict = {v: k for k, v in char_dict.items()}
    index_dict[char_dict['<space>']] = ' '
    index_dict[char_dict['<NOISE>']] = '*'
    index_dict[char_dict['<eos>']] = ''
    index_dict[char_dict['<sos>']] = ''
    beamcoder = BeamCoder(listener, attender, speller, None,
                          index_dict, char_dict['<sos>'],
                          char_dict['<eos>'], args)
    recognize_samples_results(sample_indices, test_loader, beamcoder,
                              device, index_dict)
    return


def recognize_samples_results(sample_indices, test_loader, beamcoder,
                              device, index_dict):
    # teach force decode
    for batch_index, batch_dict in enumerate(test_loader):
        if batch_index in sample_indices:
            input_utterance = batch_dict['spectrograms'].to(device)
            ground_truth = batch_dict['ground_truths'][0]
            padded_label = batch_dict['target_labels'][0]
            input_length = batch_dict['input_lengths']
            # teach force decoding
            nbest_hyps = beamcoder.beam_search_decoding(
                input_utterance, input_length, padded_label, teach_force=True)
            att_weights = nbest_hyps[0]['att_weights']
            att_weights = torch.stack(att_weights)
            plt.imshow(att_weights.cpu().detach().numpy(), cmap='rainbow')
            plt.savefig('./image/teach_force{}.png'.format(batch_index),
                        transparent=True)
            teach_force = nbest_hyps[0]['seq']
            # beam search decoding
            nbest_hyps = beamcoder.beam_search_decoding(
                input_utterance, input_length, ground_truth=None, teach_force=False)
            att_weights = nbest_hyps[0]['att_weights']
            att_weights = torch.stack(att_weights)
            plt.imshow(att_weights.cpu().detach().numpy(), cmap='rainbow')
            plt.savefig('./image/recognize{}.png'.format(batch_index),
                        transparent=True)
            recogn = nbest_hyps[0]['seq']
            # print results
            print('recognize sample {}'.format(batch_index))
            print('sample data grouth truth: ')
            print(''.join([index_dict[int(x)] for x in ground_truth]).lower())
            print('sample data teach force: ')
            print(''.join([index_dict[int(x)] for x in teach_force]).lower())
            print('sample data beam decode: ')
            print(''.join([index_dict[int(x)] for x in recogn]).lower())
        if batch_index > sample_indices[-1]:
            break
    return


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    get_recognition_results(sample_indices)
