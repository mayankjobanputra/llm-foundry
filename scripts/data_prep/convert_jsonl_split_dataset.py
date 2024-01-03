"""Local JSONL split dataset conversion script.

python scripts/data_prep/convert_jsonl_split_dataset.py
--dataset_path ~/Projects/UT/wh_multilingual/data/en/oscar_splits --lang en
--out_root oscar_en --splits train val --concat_tokens 2048
--tokenizer_path ~/Projects/UT/wh_multilingual/models/tokenizers/en_vocab.json
--eos_text '<|endoftext|>' --compression zstd

"""
import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--lang', type=str, choices=["en", "de", "all"], required=True, help='E.g."en", "de", or "all"')
    parser.add_argument('--splits', nargs='+', default=['train', 'train_small', 'val', 'val_small'])
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer_path', type=str, required=False, default=None, help="vocab.json file path")
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, required=False, default=None)

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer_path is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer_path')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str
    raw_samples: int
    truncated_samples: Union[int, None]


@dataclass
class DatasetConstants:
    chars_per_sample: int
    chars_per_token: int
    splits = {}

    def __iter__(self):
        for _, v in self.splits.items():
            yield v


oscar_en_constants = DatasetConstants(
    chars_per_sample=4830,  # Computed over validation set
    chars_per_token=4  # OpenAI estimate
)
oscar_en_constants.splits['train'] = DataSplitConstants(hf_split='train',
                                                        folder_split='train',
                                                        raw_samples=24773,
                                                        truncated_samples=None)
oscar_en_constants.splits['train_small'] = DataSplitConstants(hf_split='train',
                                                              folder_split='train_small',
                                                              raw_samples=10000,
                                                              truncated_samples=10000)
oscar_en_constants.splits['val'] = DataSplitConstants(hf_split='val',
                                                      folder_split='val',
                                                      raw_samples=750,
                                                      truncated_samples=None)
oscar_en_constants.splits['val_small'] = DataSplitConstants(hf_split='val',
                                                            folder_split='val_small',
                                                            raw_samples=3000,
                                                            truncated_samples=3000)

oscar_de_constants = DatasetConstants(
    chars_per_sample=438429,  # Computed over validation set
    chars_per_token=4  # OpenAI estimate
)
oscar_de_constants.splits['train'] = DataSplitConstants(hf_split='train',
                                                        folder_split='train',
                                                        raw_samples=5527947,
                                                        truncated_samples=None)
oscar_en_constants.splits['train_small'] = DataSplitConstants(hf_split='train',
                                                              folder_split='train_small',
                                                              raw_samples=10000,
                                                              truncated_samples=10000)
oscar_en_constants.splits['val'] = DataSplitConstants(hf_split='val',
                                                      folder_split='val',
                                                      raw_samples=30000,
                                                      truncated_samples=None)
oscar_en_constants.splits['val_small'] = DataSplitConstants(hf_split='val',
                                                            folder_split='val_small',
                                                            raw_samples=3000,
                                                            truncated_samples=3000)

CONSTS = {'oscar_en': oscar_en_constants, 'oscar_de': oscar_de_constants}


def build_hf_dataset(
    dataset_path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
) -> IterableDataset:
    """Build an IterableDataset over the Oscar data.

    Args:
        dataset_path (str): Dataset path
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use

    Returns:
        An IterableDataset.
    """

    hf_dataset = hf_datasets.load_dataset("json",
                                          data_files={'train': f'{dataset_path}/train.jsonl',
                                                      'val': f'{dataset_path}/val.jsonl'}
                                          )
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase')
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(hf_dataset=hf_dataset[split],
                                      tokenizer=tokenizer,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      no_wrap=no_wrap)
    return dataset


def _est_progress_denominator(total_samples: int, chars_per_sample: int,
                              chars_per_token: int, mode: ConcatMode,
                              max_length: int):
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length


def build_dataloader(dataset, batch_size, num_workers) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' or 'macos' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())  # type: ignore
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create OSCAR dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    if args.lang == 'en':
        dataset = "oscar_en"
    elif args.lang == 'de':
        dataset = "oscar_de"
    elif args.lang == "all":
        raise NotImplementedError
    else:
        raise ValueError(f'Language "{args.lang}" not supported.')
    dataset_constants = CONSTS[dataset]

    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'bytes'}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {'text': 'str'}

    for split_name in args.splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError:
            raise KeyError(f'Constants not defined for split {split_name}.')
        hf_split = split.hf_split
        folder_split = split.folder_split
        expected_num_samples = split.raw_samples
        truncate_num_samples = split.truncated_samples
        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        # Get samples
        dataset = build_hf_dataset(dataset_path=args.dataset_path,
                                   split=hf_split,
                                   mode=mode,
                                   max_length=args.concat_tokens,
                                   bos_text=args.bos_text,
                                   eos_text=args.eos_text,
                                   no_wrap=args.no_wrap,
                                   tokenizer=tokenizer)
        loader = build_dataloader(dataset=dataset,
                                  batch_size=512,
                                  num_workers=args.num_workers)
        samples = generate_samples(loader,
                                   truncate_num_samples=truncate_num_samples)

        if expected_num_samples is not None:
            denominator = truncate_num_samples if truncate_num_samples is not None else _est_progress_denominator(
                total_samples=expected_num_samples,
                chars_per_sample=dataset_constants.chars_per_sample,
                chars_per_token=dataset_constants.chars_per_token,
                mode=mode,
                max_length=args.concat_tokens,
            )
        else:
            denominator = None

        # Write samples
        print(f'Converting {folder_split} to MDS format...')
        print(
            f'Note that the progress bar is based on the dataset length before tokenization.'
        )
        print(f'It will finish at a value below 100% if tokenizing')
        with MDSWriter(columns=columns,
                       out=os.path.join(args.out_root, folder_split),
                       compression=args.compression) as out:
            if denominator is not None:
                for sample in tqdm(samples,
                                   desc=folder_split,
                                   total=denominator):
                    out.write(sample)
            else:
                for sample in tqdm(samples, desc=folder_split):
                    out.write(sample)


if __name__ == '__main__':
    main(parse_args())
