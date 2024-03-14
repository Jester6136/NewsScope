from transformers import TrainingArguments
from modules.model_architeture.mrc_model import MRCQuestionAnswering
from transformers import Trainer
from modules.datasets import data_loader
import numpy as np
from datasets import load_metric
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    # # none
    # model_path = r"/data2/cmdir/home/ioit104/aiavn/NewsScope/cache/mrc_model"

    #trained
    model_path = r"/data2/cmdir/home/ioit104/aiavn/NewsScope/cache/v1/checkpoint-1176"
    model = MRCQuestionAnswering.from_pretrained(model_path)
    print(model)
    print(model.config)

    train_dataset, valid_dataset, test_dataset = data_loader.get_dataloader(
        train_path='data/data_processed/train.dataset',
        valid_path='data/data_processed/valid.dataset',
        test_path='data/data_processed/test.dataset',
    )

    training_args = TrainingArguments("cache/v1",
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=40,
                                      learning_rate=1e-4,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=32,
                                      gradient_accumulation_steps=1,
                                      logging_dir='./log',
                                      logging_steps=5,
                                      label_names=['start_positions',
                                                   'end_positions',
                                                   'span_answer_ids',
                                                   'input_ids',
                                                   'words_lengths'],
                                      group_by_length=True,
                                      save_strategy="epoch",
                                      metric_for_best_model='f1',
                                      load_best_model_at_end=True,
                                      save_total_limit=2,
                                      #eval_steps=1,
                                      #evaluation_strategy="steps",
                                      evaluation_strategy="epoch",
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_loader.data_collator,
        compute_metrics=data_loader.compute_metrics
    )

    # Evaluate the model
    results = trainer.evaluate()

    print(results)
