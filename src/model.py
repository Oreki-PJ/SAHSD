import pdb
import random
import torch
from torch import nn
import transformers
from transformers import GPT2Tokenizer


class PET:
    """Wraps basic prompt methods."""

    def __init__(self, tokenizer, reader, model, device) -> None:
        self.tokenizer = tokenizer
        self.reader = reader
        self.model = model
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

        label_ids = []
        tokenize_kwargs = {}
        if isinstance(tokenizer, GPT2Tokenizer):
            tokenize_kwargs['add_prefix_space'] = True
        for label in reader.VERBALIZERS:
            label_id = tokenizer.encode(
                label, add_special_tokens=False, **tokenize_kwargs)[0]  # Force using one token
            label_ids.append(label_id)
        self.label_ids = torch.tensor(label_ids, device=device).long()

    def forward_step(self, batch, logits_key='pet_logits'):
        # Perform PET forward on MLM model and store output back
        if type(self.model) == transformers.models.bart.modeling_bart.BartForConditionalGeneration:
            batch[logits_key] = self.model(input_ids=batch['input_ids'],
                                       attention_mask=batch['attention_mask'])[0]
        else:
            batch[logits_key] = self.model(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        token_type_ids=batch['token_type_ids'])[0]

    def get_loss(self, batch, full_vocab=False, logits_key='pet_logits'):
        # Compute Cross-Entropy loss for prompt verbalizers
        assert logits_key in batch, 'logits should be pre-computed and stored in batch dict'
        masked_logits = batch[logits_key][batch['pet_flags'] == -1]
        labels = batch['pet_labels']
        if not full_vocab:
            masked_logits = masked_logits[:, self.label_ids]
            labels = batch['label_ids']
        return self.loss_fn(masked_logits, labels)

    def get_predictions(self, batch, logits_key='pet_logits'):
        # Get predicted labels
        full_logits = batch[logits_key]
        masked_logits = full_logits[batch['pet_flags'] == -1]
        masked_logits = masked_logits[:, self.label_ids]
        return masked_logits.argmax(-1).detach().cpu()
    
    def get_probs(self, batch, logits_key='pet_logits'):
        # Get predicted labels
        full_logits = batch[logits_key]
        masked_logits = full_logits[batch['pet_flags'] == -1]
        masked_logits = masked_logits[:, self.label_ids]
        masked_logits_probs = torch.nn.functional.softmax(masked_logits, dim=1)
        return masked_logits_probs.detach().cpu()


class SKIP_First(PET):

    def __init__(self, tokenizer, reader, model, device):
        super().__init__(tokenizer, reader, model, device)
        self.pattern_map = []
        self.label_map = []

        kwargs = {}
        if isinstance(tokenizer, GPT2Tokenizer):
            kwargs['add_prefix_space'] = True

        # Initialize pattern & verbalizer mapping
        curr_idx = tokenizer.vocab_size - 1
        for part in reader.PATTERN:
            if part[0] != '[':
                token_ids = tokenizer.encode(part,
                                             add_special_tokens=False,
                                             **kwargs)
                for i in token_ids:
                    self.pattern_map.append([i, curr_idx])
                    curr_idx -= 1
        for label in reader.VERBALIZERS:
            label_id = tokenizer.encode(
                label, add_special_tokens=False, **kwargs)[0]  # Force using one token
            self.label_map.append([label_id, curr_idx])
            curr_idx -= 1

        # Target token ids
        self.pattern_ids = torch.tensor([p[1] for p in self.pattern_map],
                                        device=device).long()
        self.label_ids = torch.tensor([p[1] for p in self.label_map],
                                      device=device).long()
        self._init_embedding()

    def _init_embedding(self, copy=True):
        # Get word embedding from huggingface transformer model
        w = self.model.get_input_embeddings().weight.data
        if copy:
            for old, new in self.pattern_map + self.label_map:
                w[new] = w[old]
        else:
            for _, new in self.pattern_map + self.label_map:
                max_val = w[new].abs().max()
                w[new].uniform_(-max_val, max_val)

    def _prepare_input(self, batch):
        # Replace original token ids
        ids, flags = batch['input_ids'], batch['pet_flags']
        batch_size = len(ids)
        ids[flags == 2] = self.pattern_ids.repeat(batch_size)
        batch['input_ids'] = ids
        batch['pet_labels'] = self.label_ids[batch['label_ids']]

    def forward_step(self, batch):
        self._prepare_input(batch)
        super().forward_step(batch)

class SKIP_Second(PET):
    """Wraps differentiable prompts."""

    def __init__(self, tokenizer, reader, model, device):
        super().__init__(tokenizer, reader, model, device)
        self.pattern_map = []
        self.label_map = []
        self.sentiment_pattern_map = [[2009, 30521], [2001, 30520], [103, 103], [1012, 30519]]
        self.sentiment_label_map = [[4997, 30518], [8699, 30517], [3893, 30516]]
        # ["negative", "neutral", "positive"]
        # [0, 1, 2]

        kwargs = {}
        if isinstance(tokenizer, GPT2Tokenizer):
            kwargs['add_prefix_space'] = True

        # Initialize pattern & verbalizer mapping
        # curr_idx = tokenizer.vocab_size - 1
        curr_idx = tokenizer.vocab_size - 1 - 3
        for part in reader.PATTERN:
            if part[0] != '[':
                token_ids = tokenizer.encode(part,
                                             add_special_tokens=False,
                                             **kwargs)
                for i in token_ids:
                    self.pattern_map.append([i, curr_idx])
                    curr_idx -= 1
        for label in reader.VERBALIZERS:
            label_id = tokenizer.encode(
                label, add_special_tokens=False, **kwargs)[0]  # Force using one token
            self.label_map.append([label_id, curr_idx])
            curr_idx -= 1

        # Target token ids
        self.pattern_ids = torch.tensor([p[1] for p in self.pattern_map],
                                        device=device).long()
        self.label_ids = torch.tensor([p[1] for p in self.label_map],
                                      device=device).long()
        self.sentiment_pattern_ids = torch.tensor([p[1] for p in self.sentiment_pattern_map],
                                      device=device).long()
        self.sentiment_pattern_ids_batch = torch.tensor([p[1] for p in self.sentiment_pattern_map],
                                      device=device).long()
        self._init_embedding()

    def _init_embedding(self, copy=True):
        # Get word embedding from huggingface transformer model
        w = self.model.get_input_embeddings().weight.data
        if copy:
            for old, new in self.pattern_map + self.label_map:
                w[new] = w[old]
        else:
            for _, new in self.pattern_map + self.label_map:
                max_val = w[new].abs().max()
                w[new].uniform_(-max_val, max_val)

    def _prepare_input(self, batch):
        def sentiment_ids_mapping(x):
            for p in self.sentiment_pattern_map:
                if(x == p[0]):
                    return p[1]
        # Replace original token ids
        ids, flags = batch['input_ids'], batch['pet_flags']
        batch_size = len(ids)
        ids[flags == 2] = self.pattern_ids.repeat(batch_size)
        self.sentiment_pattern_ids[2] = self.sentiment_label_map[batch['senti_pre_id'][0]][1]
        self.sentiment_pattern_ids_batch = self.sentiment_pattern_ids
        for b in range(1, batch_size):
            self.sentiment_pattern_ids[2] = self.sentiment_label_map[batch['senti_pre_id'][b]][1]
            self.sentiment_pattern_ids_batch = torch.cat((self.sentiment_pattern_ids_batch, self.sentiment_pattern_ids),0)

        ids[flags == 3] = self.sentiment_pattern_ids_batch

        batch['input_ids'] = ids
        batch['pet_labels'] = self.label_ids[batch['label_ids']]

    def forward_step(self, batch):
        self._prepare_input(batch)
        super().forward_step(batch)


def get_pet_mappers(tokenizer, reader, model, device, pet_method):
    if pet_method == 'skip_first':
        return SKIP_First(tokenizer, reader, model, device)
    if pet_method == 'skip_second':
        return SKIP_Second(tokenizer, reader, model, device)
    raise NotImplementedError('Unsupported pet method')
