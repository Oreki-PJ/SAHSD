import pdb
import torch
import numpy as np
from transformers import GPT2Tokenizer
from torch.utils.data.dataloader import DataLoader

import pandas as pd


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta=None, idx=-1, senti_pre = None):
        """
        Create a new InputExample.
        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}
        self.senti_pre = senti_pre


class Reader:
    PATTERN = []
    LABELS = []
    VERBALIZERS = []

    def load_samples(*args):
        raise NotImplementedError()

class SBICReader(Reader):
    PATTERN = ["[text_a]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['post'], label=row['label']))

        return examples
    
class SBICReader_v2(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['post'], label=row['label'], senti_pre=row['senti_pre']))

        return examples
    
class SBICReader_v2_prompt_v2(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "Verbal abuse directed at a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['post'], label=row['label'], senti_pre=row['senti_pre']))

        return examples
    
class SBICReader_v2_prompt_v3(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "Hate speech?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['post'], label=row['label'], senti_pre=row['senti_pre']))

        return examples
    
class HatexplainTwoReader(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text_joined'], label=row['label'], senti_pre=row['senti_pre']))

        return examples

class HatexplainThrReader(Reader):
    PATTERN = ["[text_a]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["normal", "offensive", "hatespeech"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text_joined'], label=row['label']))

        return examples

class HatexplainThrReader_v2(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["normal", "offensive", "hatespeech"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text_joined'], label=row['label'], senti_pre=row['senti_pre']))

        return examples

class HS18Reader(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text'], label=row['label'], senti_pre=row['senti_pre']))

        return examples

class EthosReader(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['comment'], label=row['label'], senti_pre=row['senti_pre']))

        return examples
    
class ToxiGenReader(Reader):
    PATTERN = ["[text_a]", "[sentiment_prompt]", "offensive towards a group ?", "[mask]", "."]
    LABELS = [0, 1]
    VERBALIZERS = ["No", "Yes"]

    SENTI_LABELS = [0, 1, 2]
    SENTI_VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text'], label=row['label'], senti_pre=row['senti_pre']))

        return examples

class T4SAReader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text'], label=row['label']))

        return examples

class T4SA_SBICReader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['post'], label=row['label']))

        return examples
    
class T4SA_HatexplainTwoReader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text_joined'], label=row['label']))

        return examples
    
class T4SA_HatexplainThrReader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text_joined'], label=row['label']))

        return examples
    
class T4SA_HS18Reader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text'], label=row['label']))

        return examples
    
class T4SA_EthosReader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['comment'], label=row['label']))

        return examples
    
class T4SA_ToxiGenReader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = [0, 1, 2]
    VERBALIZERS = ["negative", "neutral", "positive"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        df_data = pd.read_csv(path, index_col=0)
        for index, row in df_data.iterrows():
            examples.append(InputExample(
                     guid=index, text_a=row['text'], label=row['label']))

        return examples


def get_data_reader(task):
    task = task.lower()
    if task in ['sbic']:
        return SBICReader()
    if task in ['sbic_v2']:
        return SBICReader_v2()
    if task in ['sbic_v2_prompt_v2']:
        return SBICReader_v2_prompt_v2()
    if task in ['sbic_v2_prompt_v3']:
        return SBICReader_v2_prompt_v3()
    if task in ['t4sa_sbic']:
        return T4SA_SBICReader()
    if task in ['t4sa']:
        return T4SAReader()
    if task in ['hatexplain_two']:
        return HatexplainTwoReader()
    if task in ['t4sa_hatexplaintwo']:
        return T4SA_HatexplainTwoReader()
    if task in ['hatexplain_thr']:
        return HatexplainThrReader()
    if task in ['hatexplain_thr_v2']:
        return HatexplainThrReader_v2()
    if task in ['t4sa_hatexplainthr']:
        return T4SA_HatexplainThrReader()
    if task in ['hs18']:
        return HS18Reader()
    if task in ['t4sa_hs18']:
        return T4SA_HS18Reader()
    if task in ['ethos']:
        return EthosReader()
    if task in ['t4sa_ethos']:
        return T4SA_EthosReader()
    if task in ['toxigen']:
        return ToxiGenReader()
    if task in ['t4sa_toxigen']:
        return T4SA_ToxiGenReader()
    raise NotImplementedError(f'Unsupported task name: {task}')


def _encode(reader, sample, tokenizer, max_seq_len):
    kwargs = {'add_prefix_space': True} if isinstance(
        tokenizer, GPT2Tokenizer) else {}

    # Encode each part
    parts, n_special = [], 2
    for p in reader.PATTERN:
        if p == '[mask]':
            parts.append([tokenizer.mask_token_id])
        elif p == '[text_a]':
            parts.append(tokenizer.encode(
                sample.text_a, add_special_tokens=False, **kwargs))
        elif p == '[sentiment_prompt]':
            sentiment_prompt = ["It", "was", "[mask]", "."]

            tmp_parts = []
            for i in sentiment_prompt:
                if i == "[mask]":
                    senti_pre_verb = reader.SENTI_VERBALIZERS[sample.senti_pre]
                    tmp_parts.append(tokenizer.encode(
                        senti_pre_verb, add_special_tokens=False, **kwargs))
                else:
                    tmp_parts.extend(tokenizer.encode(
                        i, add_special_tokens=False, **kwargs))
            parts.append(tmp_parts)
        elif p == '[text_b]':
            n_special += 1
            parts.append(tokenizer.encode(
                sample.text_b, add_special_tokens=False, **kwargs))
        else:
            parts.append(tokenizer.encode(
                p, add_special_tokens=False, **kwargs))

    # Truncate
    while sum(len(x) for x in parts) > max_seq_len - n_special - 2: ## for hs18 debug max_seq_len - n_special -> max_seq_len - n_special - 2
        pdb.set_trace()
        longest = np.argmax([len(x) for x in parts])
        parts[longest].pop()

    # Concatenate
    len_seq1 = 0
    flags = [1]  # 0 = sentence; 1 = special; 2 = prompt; -1 = <mask>; 3 = sentiment_prompt
    ids = [tokenizer.cls_token_id]
    for p, real_p in zip(reader.PATTERN, parts):
        if p == '[mask]':
            flags.append(-1)
            ids.append(tokenizer.mask_token_id)
        elif p == '[text_a]':
            flags.extend([0] * len(real_p))
            ids.extend(real_p)
            flags.append(1)
            ids.append(tokenizer.sep_token_id)
            len_seq1 = len(ids)
        elif p == '[sentiment_prompt]':
            flags.extend([3] * len(real_p))
            ids.extend(real_p)
            flags.append(1)
            ids.append(tokenizer.sep_token_id)
            len_seq1 = len(ids)
        elif p == '[text_b]':
            flags.extend([0] * len(real_p))
            ids.extend(real_p)
        else:
            flags.extend([2] * len(real_p))
            ids.extend(real_p)

    # Padding to max length
    ids.append(tokenizer.sep_token_id)
    len_seq2 = len(ids)
    ids.extend([tokenizer.pad_token_id] * (max_seq_len - len_seq2))
    flags.extend([1] * (max_seq_len - len(flags)))
    att_mask = [1] * len_seq2 + [0] * (max_seq_len - len_seq2)
    seg_ids = [0] * len_seq1 + [1] * \
        (len_seq2 - len_seq1) + [0] * (max_seq_len - len_seq2)

    # Get verbalized token id
    label_id = reader.LABELS.index(sample.label)
    verbalized_id = tokenizer.encode(
        reader.VERBALIZERS[label_id], add_special_tokens=False, **kwargs)[0]  # Force using one token
    return {'input_ids': ids, 'attention_mask': att_mask,
            'token_type_ids': seg_ids, 'label_ids': label_id,
            'pet_labels': verbalized_id, 'pet_flags': flags} 

def _encode_with_senti_pre(reader, sample, tokenizer, max_seq_len):
    kwargs = {'add_prefix_space': True} if isinstance(
        tokenizer, GPT2Tokenizer) else {}

    # Encode each part
    parts, n_special = [], 2
    for p in reader.PATTERN:
        if p == '[mask]':
            parts.append([tokenizer.mask_token_id])
        elif p == '[text_a]':
            parts.append(tokenizer.encode(
                sample.text_a, add_special_tokens=False, **kwargs))
        elif p == '[sentiment_prompt]':
            sentiment_prompt = ["It", "was", "[mask]", "."]

            tmp_parts = []
            for i in sentiment_prompt:
                if i == "[mask]":
                    senti_pre_verb = reader.SENTI_VERBALIZERS[sample.senti_pre]
                    tmp_parts.extend(tokenizer.encode(
                        senti_pre_verb, add_special_tokens=False, **kwargs))
                else:
                    tmp_parts.extend(tokenizer.encode(
                        i, add_special_tokens=False, **kwargs))
            parts.append(tmp_parts)
        elif p == '[text_b]':
            n_special += 1
            parts.append(tokenizer.encode(
                sample.text_b, add_special_tokens=False, **kwargs))
        else:
            parts.append(tokenizer.encode(
                p, add_special_tokens=False, **kwargs))

    # Truncate
    while sum(len(x) for x in parts) > max_seq_len - n_special -2: ## for hs18 debug max_seq_len - n_special -> max_seq_len - n_special - 2
        longest = np.argmax([len(x) for x in parts])
        parts[longest].pop()

    # Concatenate
    len_seq1 = 0
    flags = [1]  # 0 = sentence; 1 = special; 2 = prompt; -1 = <mask>; 3 = sentiment_prompt
    ids = [tokenizer.cls_token_id]
    for p, real_p in zip(reader.PATTERN, parts):
        if p == '[mask]':
            flags.append(-1)
            ids.append(tokenizer.mask_token_id)
        elif p == '[text_a]':
            flags.extend([0] * len(real_p))
            ids.extend(real_p)
            flags.append(1)
            ids.append(tokenizer.sep_token_id)
            len_seq1 = len(ids)
        elif p == '[sentiment_prompt]':
            flags.extend([3] * len(real_p))
            ids.extend(real_p)
            flags.append(1)
            ids.append(tokenizer.sep_token_id)
            len_seq1 = len(ids)
        elif p == '[text_b]':
            flags.extend([0] * len(real_p))
            ids.extend(real_p)
        else:
            flags.extend([2] * len(real_p))
            ids.extend(real_p)

    # Padding to max length
    ids.append(tokenizer.sep_token_id)
    len_seq2 = len(ids)
    ids.extend([tokenizer.pad_token_id] * (max_seq_len - len_seq2))
    flags.extend([1] * (max_seq_len - len(flags)))
    att_mask = [1] * len_seq2 + [0] * (max_seq_len - len_seq2)
    seg_ids = [0] * len_seq1 + [1] * \
        (len_seq2 - len_seq1) + [0] * (max_seq_len - len_seq2)

    # Get verbalized token id
    label_id = reader.LABELS.index(sample.label)
    senti_pre_id = reader.LABELS.index(sample.label)
    verbalized_id = tokenizer.encode(
        reader.VERBALIZERS[label_id], add_special_tokens=False, **kwargs)[0]  # Force using one token
    # pdb.set_trace()
    return {'input_ids': ids, 'attention_mask': att_mask,
            'token_type_ids': seg_ids, 'label_ids': label_id,
            'pet_labels': verbalized_id, 'pet_flags': flags, 'senti_pre_id': senti_pre_id}



def get_data_loader(reader, path, split, tokenizer, max_seq_len, batch_size, device, shuffle=False):
    def collate_fn(samples):
        if isinstance(reader, SBICReader_v2) or isinstance(reader, SBICReader_v2_prompt_v2) \
            or isinstance(reader, SBICReader_v2_prompt_v3) or isinstance(reader, HatexplainTwoReader) \
            or isinstance(reader, HatexplainThrReader_v2) or isinstance(reader, HS18Reader) \
            or isinstance(reader, EthosReader) or isinstance(reader, ToxiGenReader):
            encoded_outputs = [
                _encode_with_senti_pre(reader, sample, tokenizer, max_seq_len) for sample in samples]
        else:
            encoded_outputs = [
                _encode(reader, sample, tokenizer, max_seq_len) for sample in samples]
        merged_outputs = {}
        for k in encoded_outputs[0].keys():
            merged_outputs[k] = torch.tensor(
                [outputs[k] for outputs in encoded_outputs], device=device).long()
        return merged_outputs

    all_samples = reader.load_samples(path, split)
    return DataLoader(all_samples, batch_size, shuffle, collate_fn=collate_fn)
