# SAHSD

This repository contains the source code for the SAHSD (SAHSD: Enhancing Hate Speech Detection through Sentiment Analysis and Few-Shot Large Language Models) method.

## Environment
- python@3.8
- Use `pip install -r requirements.txt` to install dependencies.

## How to run
* use `run_first.py` to get pre-finetuned model and sentiment knowledge injected prompts.
* use `run_for_senti.py` to get sentiment predict using pre-finetuned model.
* use `run_second.py` to run hate speech detection.
```bash
$ python run_first.py
usage: run_first.py [-h] [--config Configuration file storing all parameters]
                [--do_train]
                [--do_test]
$ python run_for_senti.py
usage: run_for_senti.py [-h] [--config Configuration file storing all parameters]
                [--do_test]
                [--seed RANDOM_SEED]
                [--train_sample_nums TRAIN_SAMPLE_NUMBERS]
$ python run_second.py
usage: run_second.py [-h] [--config Configuration file storing all parameters]
                [--do_train]
                [--do_test]
                [--seed RANDOM_SEED]
                [--train_sample_nums TRAIN_SAMPLE_NUMBERS]

* Use the corresponding bash file to run a loop with ten different random seeds and all the train number experiments
