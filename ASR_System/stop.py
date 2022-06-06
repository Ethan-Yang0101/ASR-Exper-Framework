from CTC_Attent.Model import Model
from torch.utils.data import DataLoader
from WSJDataset import data_preprocessing
from ArgParser import get_args
from WSJDataset import WSJDataset
from CTC_Attent.BeamCoder import BeamCoder
from tqdm import tqdm
import torch
import warnings


def plot_sentence_length_histogram():
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
    gt_length, pred_length = get_results(
        test_loader, beamcoder, device)
    return


def get_results(test_loader, beamcoder, device):
    gt_length, pred_length = [], []
    pbar = tqdm(total=len(test_loader))
    for batch_index, batch_dict in enumerate(test_loader):
        input_utterance = batch_dict['spectrograms'].to(device)
        input_length = batch_dict['input_lengths']
        padded_label = batch_dict['target_labels'][0]
        nbest_hyps = beamcoder.beam_search_decoding(
            input_utterance, input_length, ground_truth=None, teach_force=False)
        hyp = nbest_hyps[0]
        gt_length.append(len(padded_label)-1)
        pred_length.append(len(hyp['seq'])-2)
        if batch_index != 0 and batch_index % 10 == 0:
            pbar.update(10)
    with open('expr_data/stop.txt', 'w') as fp:
        fp.write(str(gt_length) + '\n')
        fp.write(str(pred_length) + '\n')
    return gt_length, pred_length


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    plot_sentence_length_histogram()
