from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, _Loss
from midi_processor.processor import START_IDX
# from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class TransformerLoss(CrossEntropyLoss):
    def __init__(self, ignore_index=-100, reduction='mean') -> None:
        self.reduction = reduction
        self.ignore_index = ignore_index
        super().__init__(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(torch.long)
        mask = (target != self.ignore_index).to(input.device, dtype=torch.long)
        not_masked_length = mask.to(torch.int).sum()
        input = input.permute(0, -1, -2)
        _loss = super().forward(input, target)
        _loss *= mask.to(_loss.dtype)
        return _loss.sum() / not_masked_length

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(input, target)


class SmoothCrossEntropyLoss(_Loss):
    """
    Custom loss with label smoothing and a penalty for missing Note-Off events after Note-On events.
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits
        self.note_on_idx = START_IDX['note_on']
        self.note_off_idx = START_IDX['note_off']
        self.time_shift_idx = START_IDX['time_shift']
        self.velocity_idx = START_IDX['velocity']

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        # Smooth the target distribution
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        # Standard cross-entropy loss
        ce = self.cross_entropy_with_logits(q_prime, input)

        # Apply the penalty for missing Note-Off events within 50 MIDI events
        penalty = self._apply_note_off_penalty(target)

        # Calculate final loss
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return (ce.sum() + penalty) / lengths
        elif self.reduction == 'sum':
            return ce.sum() + penalty
        else:
            raise NotImplementedError

    def _apply_note_off_penalty(self, target):
        """
        Penalty for missing Note-Off after a Note-On event within 50 events.
        """
        penalty = 0.0
        batch_size, seq_length = target.shape

        # Iterate through each sequence in the batch
        for b in range(batch_size):
            last_note_on_time = [-1] * START_IDX['note_off']
            prev_was_velocity = False
            for t in range(seq_length):
                # window = target[b, t : min(t+50, len(t)-1)]
                token = target[b, t]

                # If NoteOn event
                if self.note_on_idx <= token < self.note_off_idx:
                    # NoteON event multiple times before NoteOff
                    if last_note_on_time[token] is not -1:
                        penalty += 1.0
                    last_note_on_time[token] = t
                
                # If NoteOff event
                if self.note_off_idx <= token < self.time_shift_idx:
                    # NoteOff event without NoteOn
                    if last_note_on_time[token - self.note_off_idx] == -1:
                        penalty += 1.0
                    # NoteOn event was > 50 tokens ago
                    elif t - last_note_on_time[token - self.note_off_idx] > 50:
                        penalty += 1.0
                    last_note_on_time[token - self.note_off_idx] = -1

                # Timeshift = 0
                if token == 256:
                    penalty += 1.0

                # Two immediately following velocity shifts
                if self.velocity_idx <= token:
                    if prev_was_velocity:
                        penalty += 1.0
                    else:
                        prev_was_velocity = True
                else:
                    prev_was_velocity = False

                # # Detect Note-On event
                # if self.note_on_idx <= token < self.note_off_idx:
                #     note_on_positions.append(t)

                # # Apply penalty if there's no corresponding Note-Off within 50 tokens
                # if len(note_on_positions) > 0:
                #     if token >= self.note_off_idx and token < self.time_shift_idx and token == note_on_positions[0] + START_IDX['note off']:  # Note-Off event
                #         note_on_positions.pop(0)  # Remove first Note-On as it is paired
                #     elif t - note_on_positions[0] > 50:  # Check if 50 tokens have passed
                #         penalty += 1.0  # Add penalty for each unpaired Note-On
                #         note_on_positions.pop(0)  # Remove the unpaired Note-On event
        
        return penalty

    def cross_entropy_with_logits(self, p, q):
        """
        Standard cross-entropy calculation with logits.
        """
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)



class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps

        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.d_model ** (-0.5) * min(arg1, arg2)

