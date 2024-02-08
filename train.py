from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer,default_data_collator
from model import model,feature_extractor,tokenizer
from dataloader import train_ds,test_ds


training_args = Seq2SeqTrainingArguments(
    output_dir="image-caption-generator", # name of the directory to store training outputs
    evaluation_strategy="epoch",          # evaluate after each epoch
    per_device_train_batch_size=8,        # batch size during training
    per_device_eval_batch_size=8,         # batch size during evaluation
    learning_rate=5e-5,
    weight_decay=0.01,                    # weight decay for AdamW optimizer
    num_train_epochs=5,                   # number of epochs to train
    save_strategy='epoch',                # save checkpoints after each epoch
    report_to='none',                     # prevents logging to wandb, mlflow...
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    data_collator=default_data_collator,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=training_args,
)

if __name__=="__main__":
    trainer.train()