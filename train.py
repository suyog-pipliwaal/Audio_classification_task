import torch
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import joblib
import os
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, AutoFeatureExtractor
os.environ['WANDB_MODE'] = 'disabled'
def load_data(h5_file_path, test_size=0.2, random_state=42):
    with h5py.File(h5_file_path, 'r') as f:
        log_melspec = np.array(f['log_melspecs'])
        label = np.array(f['labels'])
    print(f"Loaded {log_melspec.shape[0]} samples from {h5_file_path}")

    X_train, X_val, y_train, y_val = train_test_split(log_melspec, label, test_size=test_size, random_state=random_state, stratify=label)

    print(f"Train Samples: {X_train.shape[0]}, Validation Samples: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, feature_extractor):
        self.X = X
        self.y = y
        self.feature_extractor = feature_extractor
    def __getitem__(self, index):
        log_melSpec = torch.tensor(self.X[index], dtype=torch.float32)
        label = torch.tensor(self.y[index], dtype=torch.int64)
        inputs = self.feature_extractor(log_melSpec, sampling_rate=16000,  return_tensors="pt")
        return {"input_values":inputs['input_values'].squeeze(), "labels": label}
    def __len__(self):
        return len(self.X)
    

if __name__ == '__main__':
    TRAIN           = './train.h5'
    MODEL_NAME      = "facebook/wav2vec2-base"
    label_encoder   = joblib.load("./label_encoder.pkl")
    class_names     = label_encoder.classes_
    id2label        = {i: label for i, label in enumerate(class_names)},
    label2id        = {label: i for i, label in enumerate(class_names)}
    # print(class_names, id2label, label2id)
    num_labels      = len(id2label)
    model           = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    X_train, X_val, y_train, y_val  = load_data(TRAIN)
    train_dataset   = AudioDataset(X_train, y_train, feature_extractor)
    val_dataset     = AudioDataset(X_val, y_val, feature_extractor)

    training_args = TrainingArguments(
            output_dir="./models",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=1e-5,
            logging_dir="./logs",
            logging_steps=100,
            report_to="none",  # Disable Weights & Biases logging
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        # compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    # Save Model
    trainer.save_model("saved_model")