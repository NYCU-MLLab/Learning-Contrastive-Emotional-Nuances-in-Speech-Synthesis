import torch

from torch import nn
from info_nce import InfoNCE, info_nce
import random
from daft_exprt.data_loader import prepare_data_loaders
import numpy as np

class DaftExprtLoss(nn.Module):
    def __init__(self, gpu, hparams):
        super(DaftExprtLoss, self).__init__()
        self.nb_channels = hparams.n_mel_channels
        self.warmup_steps = hparams.warmup_steps
        self.adv_max_weight = hparams.adv_max_weight
        self.post_mult_weight = hparams.post_mult_weight
        self.dur_weight = hparams.dur_weight
        self.energy_weight = hparams.energy_weight
        self.pitch_weight = hparams.pitch_weight
        self.mel_spec_weight = hparams.mel_spec_weight
        self.contr_weight = hparams.contr_weight
        self.emotion_classifier_loss = hparams.emotion_classifier_loss
        
        self.L1Loss = nn.L1Loss(reduction='none').cuda(gpu)
        self.MSELoss = nn.MSELoss(reduction='none').cuda(gpu)
        self.CrossEntropy = nn.CrossEntropyLoss().cuda(gpu)
        self.InfoEnce = InfoNCE(negative_mode='paired', temperature=0.01)
        self.hparams = hparams
        self.gpu = gpu
        self.use_emotion_classifier = hparams.use_emotion_classifier
        if hparams.use_emotion_classifier:
            self.mapping = {emotion: i + 1 for i, emotion in enumerate(hparams.emotions)}
        # train_loader, train_sampler, val_loader, self.emotion_loaders ,nb_training_examples = \
        # prepare_data_loaders(hparams, num_workers=8, emotion=hparams.emotion, emotions=hparams.emotions)
        # self.emotion_iters = iter(self.emotion_loaders)
    
    def update_adversarial_weight(self, iteration):
        ''' Update adversarial weight value based on iteration
        '''
        weight_iter = iteration * self.warmup_steps ** -1.5 * self.adv_max_weight / self.warmup_steps ** -0.5
        weight = min(self.adv_max_weight, weight_iter)
        
        return weight

    def contrastive_loss_self_supervised(self, prosody_embeds, ss_batch, inputs, current_model, gpu):
        bs = prosody_embeds.size(0)
        positive_pairs = []
        negative_pairs = [ [] for _ in range(bs) ]
        ss_batch = [list(row) for row in zip(*ss_batch)]
        inputs = [list(row) for row in zip(*inputs)]
        neg_flag = 0
        for i in range(self.hparams.batch_size):
            # current_emotion = emotions[i]
            if self.hparams.contrastive_type == "self-supervised":
                mel = inputs[i][8]
                alpha = np.random.uniform(low=0.9, high=1.1)
                inputs[i][8] = vtlp(mel, self.hparams.sampling_rate, alpha, self.gpu)
                positive_pairs.append(inputs[i][0:11] + [inputs[i][13]])
            
            # Negative pairs     
            for j in range(neg_flag*8, neg_flag*8+8):
                negative_pairs[i].append(ss_batch[j][0:11] + [ss_batch[j][13]])
            neg_flag+=1

        with torch.no_grad():
            positive_pairs = current_model(self.collate_pairs(positive_pairs, gpu), emo=True)
            for i in range(self.hparams.batch_size):
                negative_pairs[i] = current_model(self.collate_pairs(negative_pairs[i], gpu), emo=True)
        pos = torch.stack([t.clone().detach() for t in positive_pairs])
        neg = torch.stack([t.clone().detach() for t in negative_pairs])
        return self.InfoEnce(prosody_embeds, pos, neg)
        
    def contrastive_loss(self, prosody_embeds, emotions, current_model, emotion_batch, gpu):
        bs = prosody_embeds.size(0)
        positive_pairs = []
        negative_pairs = [ [] for _ in range(bs) ]
        for e in emotion_batch:
            emotion_batch[e] = [list(row) for row in zip(*emotion_batch[e])]

        for i in range(len(emotions)):
            current_emotion = emotions[i]
            positive_pairs.append(emotion_batch[current_emotion][i][0:11] + [emotion_batch[current_emotion][i][13]])
            
            # Negative pairs     
            for other_emotion in emotion_batch.keys():
                if other_emotion != current_emotion:
                    # Take 1 wav from other emotion
                    random_numbers = [random.randint(0, bs-1) for _ in range(1)]
                    for j in (random_numbers):
                        try:
                            negative_pairs[i].append(emotion_batch[other_emotion][j][0:11] + [emotion_batch[other_emotion][j][13]])
                        except IndexError as e:
                            print(f"IndexError: {e}. Could not access the required elements in emotion_batch.")
                        except Exception as ex:
                            print(f"An unexpected error occurred: {ex}")

        with torch.no_grad():
            positive_pairs = current_model(self.collate_pairs(positive_pairs, gpu), emo=True)
            for i in range(len(emotions)):
                negative_pairs[i] = current_model(self.collate_pairs(negative_pairs[i], gpu), emo=True)
        pos = torch.stack([t.clone().detach() for t in positive_pairs])
        neg = torch.stack([t.clone().detach() for t in negative_pairs])
        return self.InfoEnce(prosody_embeds, pos, neg)
      
    def collate_pairs(self, batch, gpu):
        _, ids_sorted_decreasing = \
            torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        
        speaker_ids = torch.LongTensor(len(batch))
        
        for i in range(len(ids_sorted_decreasing)):
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][10]
        
        # find mel-spec max length
        max_output_len = max([x[8].size(1) for x in batch])
        
        # right zero-pad mel-specs to max output length
        frames_energy = torch.FloatTensor(len(batch), max_output_len).zero_()
        frames_pitch = torch.FloatTensor(len(batch), max_output_len).zero_()
        mel_specs = torch.FloatTensor(len(batch), self.hparams.n_mel_channels, max_output_len).zero_()
        output_lengths = torch.LongTensor(len(batch))
        emotions = []
        
        for i in range(len(ids_sorted_decreasing)):
            # extract batch sequences
            frames_energy_seq = batch[ids_sorted_decreasing[i]][6]
            frames_pitch_seq = batch[ids_sorted_decreasing[i]][7]
            mel_spec = batch[ids_sorted_decreasing[i]][8]
            # fill padded arrays
            frames_energy[i, :frames_energy_seq.size(0)] = frames_energy_seq
            frames_pitch[i, :frames_pitch_seq.size(0)] = frames_pitch_seq
            mel_specs[i, :, :mel_spec.size(1)] = mel_spec
            output_lengths[i] = mel_spec.size(1)
            
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][10]
            
            emotions.append(batch[ids_sorted_decreasing[i]][11])
        
        frames_energy = frames_energy.cuda(gpu, non_blocking=True).float()           # (B, T_max)
        frames_pitch = frames_pitch.cuda(gpu, non_blocking=True).float()             # (B, T_max)
        mel_specs = mel_specs.cuda(gpu, non_blocking=True).float()                   # (B, n_mel_channels, T_max)
        output_lengths = output_lengths.cuda(gpu, non_blocking=True).long()          # (B, )
        speaker_ids = speaker_ids.cuda(gpu, non_blocking=True).long()                # (B, )
        
        return frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, emotions, None
            
    def forward(self, inputs, outputs, targets, iteration, current_model = None, emotion_batch = None, ss_batch = None, gpu = None):
        ''' Compute training loss

        :param outputs:         outputs predicted by the model
        :param targets:         ground-truth targets
        :param iteration:       current training iteration
        '''
        # extract ground-truth targets
        # targets are already zero padded
        duration_targets, energy_targets, pitch_targets, mel_spec_targets, speaker_ids, emotions, mapped_emotions = targets
        duration_targets.requires_grad = False
        energy_targets.requires_grad = False
        pitch_targets.requires_grad = False
        mel_spec_targets.requires_grad = False
        speaker_ids.requires_grad = False
        if mapped_emotions is not None:
           mapped_emotions.requires_grad = False
        
        # extract predictions
        # predictions are already zero padded
        speaker_preds, film_params, encoder_preds, decoder_preds, _, mi, prosody_embeds, emotion, emotion_preds = outputs
        # speaker_preds, film_params, encoder_preds, decoder_preds, _, mi = outputs
        post_multipliers, _, _, _ = film_params
        duration_preds, energy_preds, pitch_preds, input_lengths = encoder_preds
        mel_spec_preds, output_lengths= decoder_preds
        
        # compute adversarial speaker objective
        speaker_loss = self.CrossEntropy(speaker_preds, speaker_ids)
        emotion_loss = 0.0
        if self.use_emotion_classifier:
            emotion_loss = self.CrossEntropy(emotion_preds, mapped_emotions)
        # compute L2 penalized loss on FiLM scalar post-multipliers
        if self.post_mult_weight != 0.:
            post_mult_loss = torch.norm(post_multipliers, p=2)
        else:
            post_mult_loss = torch.tensor([0.]).cuda(speaker_loss.device, non_blocking=True).float()
        
        # compute duration loss
        duration_loss = self.MSELoss(duration_preds, duration_targets)  # (B, L_max)
        # divide by length of each sequence in the batch
        duration_loss = torch.sum(duration_loss, dim=1) / input_lengths  # (B, )
        duration_loss = torch.mean(duration_loss)
        
        # compute energy loss
        energy_loss = self.MSELoss(energy_preds, energy_targets)  # (B, L_max)
        # divide by length of each sequence in the batch
        energy_loss = torch.sum(energy_loss, dim=1) / input_lengths  # (B, )
        energy_loss = torch.mean(energy_loss)
        
        # compute pitch loss
        pitch_loss = self.MSELoss(pitch_preds, pitch_targets)  # (B, L_max)
        # divide by length of each sequence in the batch
        pitch_loss = torch.sum(pitch_loss, dim=1) / input_lengths  # (B, )
        pitch_loss = torch.mean(pitch_loss)
        
        # compute mel-spec loss
        mel_spec_l1_loss = self.L1Loss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        mel_spec_l2_loss = self.MSELoss(mel_spec_preds, mel_spec_targets)  # (B, n_mel_channels, T_max)
        # divide by length of each sequence in the batch
        mel_spec_l1_loss = torch.sum(mel_spec_l1_loss, dim=(1, 2)) / (self.nb_channels * output_lengths)  # (B, )
        mel_spec_l1_loss = torch.mean(mel_spec_l1_loss)
        mel_spec_l2_loss = torch.sum(mel_spec_l2_loss, dim=(1, 2)) / (self.nb_channels * output_lengths)  # (B, )
        mel_spec_l2_loss = torch.mean(mel_spec_l2_loss)

        contrastive_loss = 0.0
        if emotion_batch is not None and emotion_batch != {} and self.hparams.contrastive_type == "supervised":
            contrastive_loss = self.contrastive_loss(prosody_embeds, emotion, current_model, emotion_batch, gpu)
        elif self.hparams.contrastive_type == "self-supervised" and ss_batch is not None:
            contrastive_loss = self.contrastive_loss_self_supervised(prosody_embeds, ss_batch, inputs, current_model, gpu)
        # add weights
        speaker_weight = self.update_adversarial_weight(iteration)
        speaker_loss = speaker_weight * speaker_loss
        post_mult_loss = self.post_mult_weight * post_mult_loss
        duration_loss = self.dur_weight * duration_loss
        energy_loss = self.energy_weight * energy_loss
        pitch_loss = self.pitch_weight * pitch_loss
        mel_spec_l1_loss = self.mel_spec_weight * mel_spec_l1_loss
        mel_spec_l2_loss = self.mel_spec_weight * mel_spec_l2_loss
        contrastive_loss = self.contr_weight * contrastive_loss
        emotion_classifier_loss = self.emotion_classifier_loss * emotion_loss
        
        loss = speaker_loss + post_mult_loss + duration_loss + energy_loss + pitch_loss + mel_spec_l1_loss + mel_spec_l2_loss + contrastive_loss + emotion_classifier_loss

        # create individual loss tracker
    
        individual_loss = {'speaker_loss': speaker_loss.item(), 'post_mult_loss': post_mult_loss.item(),
                           'duration_loss': duration_loss.item(), 'energy_loss': energy_loss.item(), 'pitch_loss': pitch_loss.item(),
                           'mel_spec_l1_loss': mel_spec_l1_loss.item(), 'mel_spec_l2_loss': mel_spec_l2_loss.item(),
                           'contrastive_loss': contrastive_loss.item() if (emotion_batch is not None and emotion_batch != {}) or (ss_batch is not None and ss_batch is not {}) else contrastive_loss,
                           'emotion_classifier_loss': emotion_classifier_loss.item() if self.use_emotion_classifier else emotion_classifier_loss
                           }
        
        return loss, individual_loss


def vtlp(x, fs, alpha, gpu):
    # S = stft(x).T
    S = x.cpu().numpy().T
    T, K = S.shape
    dtype = S.dtype

    f_warps = warp_freq(K, fs, alpha=alpha)
    f_warps *= (K - 1)/max(f_warps)
    new_S = np.zeros([T, K], dtype=dtype)

    for k in range(K):
        # first and last freq
        if k == 0 or k == K-1:
            new_S[:, k] += S[:, k]
        else:
            warp_up = f_warps[k] - np.floor(f_warps[k])
            warp_down = 1 - warp_up
            pos = int(np.floor(f_warps[k]))

            new_S[:, pos] += warp_down * S[:, k]
            new_S[:, pos+1] += warp_up * S[:, k]

    # y = istft(new_S.T)
    # y = fix_length(y, T)
    res = torch.from_numpy(new_S.T)
    return res.cuda(gpu, non_blocking=True).float()

def warp_freq(n_fft, fs, fhi=4800, alpha=0.9):
    bins = np.linspace(0, 1, n_fft)
    f_warps = []

    scale = fhi * min(alpha, 1)
    f_boundary = scale / alpha
    fs_half = fs // 2

    for k in bins:
        f_ori = k * fs
        if f_ori <= f_boundary:
            f_warp = f_ori * alpha
        else:
            f_warp = fs_half - (fs_half - scale) / (fs_half - scale / alpha) * (fs_half - f_ori)
        f_warps.append(f_warp)

    return np.array(f_warps)