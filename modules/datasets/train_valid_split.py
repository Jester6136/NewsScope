import datasets
import argparse
import logging
from datasets import concatenate_datasets
import glob

# datasets.temp_seed(101)
datasets.disable_progress_bar()


def assert_sample(sample):
    assert sample['context'][sample['answer_start_idx']: sample['answer_start_idx'] + len(sample['answer_text'])] == \
           sample['answer_text'], sample
    assert len(sample['context']) > 0
    assert len(sample['question']) > 0
    return True


def format_sample(sample):
    context_prev = sample['context'][:sample['answer_start_idx']].split()
    sample['answer_word_start_idx'] = len(context_prev)
    sample['answer_word_end_idx'] = len(context_prev) + len(sample['answer_text'].split()) - 1
    return sample


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_in",
                        default=r"data/data_processed/squad_mrc.jsonl",
                        type=str,
                        help="")

    args = parser.parse_args()
    train_set = []
    valid_set = []
    part = args.file_in
    dataset = datasets.load_dataset('json', data_files=[part])['train']
    dataset.filter(assert_sample)
    dataset = dataset.map(format_sample)

    all_data = dataset.train_test_split(test_size=0.1)
    train = all_data['train']
    valid = all_data['test']
    train_set.append(train)
    valid_set.append(valid)

    train_dataset = concatenate_datasets(train_set)
    valid_dataset = concatenate_datasets(valid_set)

    train_dataset.save_to_disk('data/data_processed/train.dataset')
    valid_dataset.save_to_disk('data/data_processed/valid.dataset')

    logger.info("Train: {} samples".format(len(train_dataset)))
    logger.info("Valid: {} samples".format(len(valid_dataset)))
