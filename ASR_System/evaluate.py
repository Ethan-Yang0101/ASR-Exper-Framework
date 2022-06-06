from CTC_Attent.Model import Model
from CTC_Attent.BeamCoder import BeamCoder
from torch.utils.data import DataLoader
from WSJDataset import data_preprocessing
from ArgParser import get_args
from WSJDataset import WSJDataset
import Levenshtein
from tqdm import tqdm
import torch
import warnings


def evaluate_recognition_results():
    # load argument parser
    args = get_args()
    # load dataset
    test_dataset = WSJDataset.make_speech_dataset(
        char_filename=args.char_filename,
        data_scp_filename=args.test_data_scp_filename,
        label_scp_filename=args.test_label_scp_filename,
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
    # create a model
    CTC_Attent_model = Model.load_model(args.checkpoint_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CTC_Attent_model.to(device)
    CTC_Attent_model.eval()
    listener = CTC_Attent_model.listener
    speller = CTC_Attent_model.speller
    attender = CTC_Attent_model.attender
    # create a beam search decoder
    char_dict = test_dataset.char_dict
    index_map = {v: k for k, v in char_dict.items()}
    index_map[char_dict['<space>']] = ' '
    index_map[char_dict['<NOISE>']] = '*'
    index_map[char_dict['<sos>']] = ''
    index_map[char_dict['<eos>']] = ''
    beamcoder = BeamCoder(listener=listener,
                          attender=attender,
                          speller=speller,
                          language_model=None,
                          index_map=index_map,
                          sos_id=char_dict['<sos>'],
                          eos_id=char_dict['<eos>'],
                          args=args)
    # compute word error rate and character error rate
    running_cer = 0.0
    running_wer = 0.0
    pbar = tqdm(total=len(test_loader))
    for batch_index, batch_dict in enumerate(test_loader):
        # decode an utterance
        input_utterance = batch_dict['spectrograms'].to(device)
        ground_truth = batch_dict['ground_truths'][0]
        ground_truth = ''.join([index_map[int(x)] for x in ground_truth])
        input_length = batch_dict['input_lengths']
        nbest_hyps = beamcoder.beam_search_decoding(
            input_utterance, input_length, ground_truth=None, teach_force=False)
        hyp = nbest_hyps[0]
        prediction = ''.join([index_map[int(x)] for x in hyp['seq']])
        cer_t = compute_character_error_rate(ground_truth, prediction)
        wer_t = compute_word_error_rate(ground_truth, prediction)
        running_cer += (cer_t - running_cer) / (batch_index + 1)
        running_wer += (wer_t - running_wer) / (batch_index + 1)
        if batch_index != 0 and batch_index % 10 == 0:
            pbar.update(10)
            pbar.set_postfix_str(
                " cer: {:.4f}, wer: {:.4f}".format(running_cer, running_wer))
    message = "character error rate: {:.4f}\n".format(running_cer)
    pbar.write(message)
    message = "word error rate: {:.4f}\n".format(running_wer)
    pbar.write(message)
    return


def compute_character_error_rate(ground_truth, prediction):
    if ground_truth == '' or prediction == '':
        return 0.0
    error = Levenshtein.distance(ground_truth, prediction)
    return error / len(ground_truth)


def compute_word_error_rate(ground_truth, prediction):
    ground_truth = ground_truth.split(' ')
    prediction = prediction.split(' ')
    if not ground_truth or not prediction:
        return 0.0
    error = Levenshtein.distance(ground_truth, prediction)
    return error / len(ground_truth)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    evaluate_recognition_results()
