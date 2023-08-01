python scripts/train/train_tokenizer.py --json_input_dir ~/Projects/UT/wh_multilingual/data/en/oscar_splits/ --tokenizer_output_path ~/Projects/UT/wh_multilingual/models/tokenizers/en_vocab.json --vocab_size 20000

python scripts/train/train_tokenizer.py --json_input_dir ~/Projects/UT/wh_multilingual/data/de/oscar_splits/ --tokenizer_output_path ~/Projects/UT/wh_multilingual/models/tokenizers/de_vocab.json --vocab_size 20000

python scripts/train/train_tokenizer.py --json_input_dir ~/Projects/UT/wh_multilingual/data/en/oscar_nonq_splits/ --tokenizer_output_path ~/Projects/UT/wh_multilingual/models/tokenizers/en_nonq_vocab.json --vocab_size 20000
