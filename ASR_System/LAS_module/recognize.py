from LAS_Model.LAS_Model import LAS_Model
from LAS_Model.LAS_BeamCoder import LAS_BeamCoder
from torch.utils.data import DataLoader
from Joint_module.WSJDataset import data_preprocessing
from Joint_module.ArgParser import get_args
from Joint_module.WSJDataset import WSJDataset
import matplotlib.pyplot as plt
import torch
import warnings


def get_recognition_results(sample_indices):
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
                             shuffle=True,
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
    recognize_samples_results(sample_indices, test_loader,
                              LAS_model, LAS_beamcoder,
                              device, index_dict, args)
    return


def recognize_samples_results(sample_indices, test_loader, LAS_model, LAS_beamcoder,
                              device, index_dict, args):
    # teach force decode
    for batch_index, batch_dict in enumerate(test_loader):
        if batch_index in sample_indices:
            padded_input = batch_dict['spectrograms'].to(device)
            padded_target = batch_dict['target_sequences'].to(device)
            padded_label = batch_dict['target_labels'].to(device)
            input_lengths = batch_dict['input_lengths']
            target_lengths = batch_dict['target_lengths']
            if args.apply_ctc_task:
                _, _, _, att_weights = LAS_model(
                    padded_input, input_lengths, padded_target,
                    padded_label, target_lengths)
            else:
                _, att_weights = LAS_model(
                    padded_input, input_lengths, padded_target,
                    padded_label, target_lengths)
            att_weights = att_weights.squeeze()
            plt.imshow(att_weights.cpu().detach().numpy(), cmap='rainbow')
            plt.savefig('./image/teach_force{}.png'.format(batch_index))
            # decode an utterance
            input_utterance = batch_dict['spectrograms'].to(device)
            target_label = batch_dict['target_labels'][0][:-1]
            print('recognize sample {}'.format(batch_index))
            print('sample data grouth truth: ')
            print(''.join([index_dict[int(x)] for x in target_label]).lower())
            input_length = batch_dict['input_lengths']
            nbest_hyps = LAS_beamcoder.beam_search_decoding(
                input_utterance, input_length)
            print('recognition results: ')
            for hyp in nbest_hyps:
                message = ''.join([index_dict[int(x)]
                                   for x in hyp['seq'][1:-1]]).lower()
                print(message)
            for hyp in nbest_hyps:
                message = ''.join([index_dict[int(x)]
                                   for x in hyp['seq'][1:-1]]).lower()
                if message == '':
                    continue
                att_weights = hyp['att_weights']
                att_weights = torch.stack(att_weights).squeeze()
                plt.imshow(att_weights.cpu().detach().numpy(), cmap='rainbow')
                plt.savefig('./image/recognize{}.png'.format(batch_index))
                break
        if batch_index > sample_indices[-1]:
            break
    return


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    sample_indices = [5, 10, 15, 20, 25]
    get_recognition_results(sample_indices)
