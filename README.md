# hFT-Transformer

This repository forks **"Automatic Piano Transcription with Hierarchical Frequency-Time Transformer"** presented in ISMIR2023 ([arXiv 2307.04305](https://arxiv.org/abs/2307.04305)).
and implements an inference script to transcribe directories of piano files.

## Usage
For training and evaluation, see the original repo.

The inference script we built uses the model trained for MAESTRO. We haven't yet evaluated it on different datasets, we just ran it.

To run, download `model_016_003.pkl`:

```
$ wget https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip
$ unzip checkpoint.zip
```

Then, put the files you want to transcribe in a directory, `<input_dir>`. They can be `.wav` or `.mp3` files.

```
python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe <input_dir> \
    -output_dir <output_dir> \
    -f_config corpus/MAESTRO-V3/dataset/config.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model_016_003.pkl
```    

## Development Environment
- OS
  + Ubuntu 18.04
- memory
  + 32GB
- GPU
  + corpus generation, evaluation
    - NVIDIA GeForce RTX 2080 Ti
  + training
    - NVIDIA A100
- Python
  + 3.6.9
- Required Python libraries
  + [requirements.txt](requirements.txt)

## Citation
Keisuke Toyama, Taketo Akama, Yukara Ikemiya, Yuhta Takida, Wei-Hsiang Liao, and Yuki Mitsufuji, "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer," in Proceedings of the 24th International Society for Music Information Retrieval Conference, 2023.
```
@inproceedings{toyama2023,
    author={Keisuke Toyama and Taketo Akama and Yukara Ikemiya and Yuhta Takida and Wei-Hsiang Liao and Yuki Mitsufuji},
    title={Automatic Piano Transcription with Hierarchical Frequency-Time Transformer},
    booktitle={Proceedings of the 24th International Society for Music Information Retrieval Conference},
    year={2023}
}
```
