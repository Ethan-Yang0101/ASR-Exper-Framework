from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch
import os


class ModelTrainer(object):

    def __init__(self, dataloaders, model, optimizer, args):
        self.train_loader = dataloaders['train_loader']
        self.valid_loader = dataloaders['valid_loader']
        self.test_loader = dataloaders['test_loader']
        self.model = model
        self.optimizer = optimizer
        # hyperparameters
        self.checkpoint_model_path = args.checkpoint_model_path
        self.save_folder_path = args.save_folder_path
        self.save_checkpoint = args.save_checkpoint
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.apply_ctc_task = args.apply_ctc_task
        self.use_sampling = args.use_sampling
        self.sampling_rate = args.sampling_rate
        self.reset_train_setting()

    def make_train_state(self):
        return {'epoch_index': [],
                'train_loss': [],
                'val_loss': [],
                'test_loss': [],
                'train_ctc_loss': [],
                'train_att_loss': [],
                'val_ctc_loss': [],
                'val_att_loss': []}

    def reset_train_setting(self):
        os.makedirs(self.save_folder_path, exist_ok=True)
        self.train_state = self.make_train_state()
        self.start_epoch = 0
        self.lr_scheduler = ExponentialLR(self.optimizer, 0.9, verbose=False)
        if self.checkpoint_model_path:
            package = torch.load(self.checkpoint_model_path)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.train_state = package['train_state']
            self.lr_scheduler = ExponentialLR(
                self.optimizer, 0.9,
                last_epoch=self.train_state['epoch_index'][-1],
                verbose=False)
            self.start_epoch = self.train_state['epoch_index'][-1] + 1
            print('Loading checkpoint model %s from epoch %d\n' %
                  (self.checkpoint_model_path,
                   self.train_state['epoch_index'][-1]))
        return

    def save_checkpoint_model(self, epoch_index):
        if self.save_checkpoint:
            file_path = os.path.join(
                self.save_folder_path, 'epoch%d.pth.tar' % epoch_index)
            torch.save(self.model.serialize(
                self.model, self.optimizer, self.train_state), file_path)
            print('\nSaving checkpoint model to %s\n' % file_path)
        return

    def train_val_test_model(self):
        try:
            for epoch_index in range(self.start_epoch, self.num_epochs):
                self.train_state['epoch_index'].append(epoch_index)
                self.run_one_epoch('train')
                self.lr_scheduler.step()
                self.run_one_epoch('eval')
                self.save_checkpoint_model(epoch_index)
            self.run_one_epoch('test')
        except KeyboardInterrupt:
            print("\nInterrupt, Exiting loop!")

    def run_one_epoch(self, train_mode):
        loss_type = None
        if train_mode == 'train':
            dataloader = self.train_loader
            self.model.train()
            loss_type = 'train_loss'
        if train_mode == 'eval':
            dataloader = self.valid_loader
            self.model.eval()
            loss_type = 'val_loss'
        if train_mode == 'test':
            dataloader = self.test_loader
            self.model.eval()
            loss_type = 'test_loss'
        running_loss, run_att_loss, run_ctc_loss = 0.0, 0.0, 0.0
        pbar = tqdm(total=len(dataloader))
        for batch_index, batch_dict in enumerate(dataloader):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            padded_input = batch_dict['spectrograms'].to(device)
            padded_target = batch_dict['target_sequences'].to(device)
            padded_label = batch_dict['target_labels'].to(device)
            padded_truth = batch_dict['ground_truths'].to(device)
            input_lengths = batch_dict['input_lengths']
            ground_lengths = batch_dict['ground_lengths']
            loss_t, att_loss, ctc_loss, _ = self.model(
                padded_input, input_lengths, padded_target,
                padded_label, padded_truth, ground_lengths,
                self.use_sampling, self.sampling_rate)
            run_att_loss += (att_loss.item() -
                             run_att_loss) / (batch_index + 1)
            run_ctc_loss += (ctc_loss.item() -
                             run_ctc_loss) / (batch_index + 1)
            if train_mode == 'train':
                self.optimizer.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            running_loss += (loss_t.item() - running_loss) / (batch_index + 1)
            if batch_index != 0 and batch_index % 5 == 0:
                pbar.update(5)
                pbar.set_postfix_str(
                    " loss: {:.4f}, att_loss: {:.4f}, ctc_loss: {:.4f}".format(
                        running_loss, run_att_loss, run_ctc_loss))
        self.train_state[loss_type].append(running_loss)
        if train_mode == 'train':
            self.train_state['train_att_loss'].append(run_att_loss)
            self.train_state['train_ctc_loss'].append(run_ctc_loss)
        if train_mode == 'eval':
            self.train_state['val_att_loss'].append(run_att_loss)
            self.train_state['val_ctc_loss'].append(run_ctc_loss)
        pbar.write("Epoch: {} / {} -- {} Loss: {:.4f}".format(
            self.train_state['epoch_index'][-1], self.num_epochs - 1,
            train_mode.capitalize(), self.train_state[loss_type][-1]))
