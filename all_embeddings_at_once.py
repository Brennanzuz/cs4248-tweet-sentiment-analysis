# %% [markdown]
# # Transformer model

# %%
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, TFAutoModel, RobertaTokenizerFast, TFRobertaModel, TFXLNetModel, XLNetTokenizer
import tensorflow as tf
import tf_keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.layers import *
from tf_keras.models import Model, Sequential
from tf_keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tf_keras.utils import to_categorical
from tf_keras.models import load_model

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from keras.models import load_model

# Load pre-trained embeddings
import gensim.downloader as api
glove_vectors = api.load("glove-wiki-gigaword-300")

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# ## Check GPU

# %%
# Check if TensorFlow can see GPU
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# More detailed GPU information
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("GPU name:", gpu.name)
        print("GPU details:", tf.config.experimental.get_device_details(gpu))
else:
    print("No GPU detected. TensorFlow is running on CPU.")

# Simple test to confirm GPU operation
if gpus:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c)
        print("Executed on GPU")

# %% [markdown]
# ## Load and split data

# %%
train_merged = pd.read_csv("data/train_merged.csv")
train_merged["cleaned_text"] = train_merged["cleaned_text"].fillna("")
train_merged = train_merged.fillna(0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_merged["sentiment"].values)
x = train_merged.drop("sentiment", axis = 1)
x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, random_state=42, test_size=0.2, shuffle=True)

# %% [markdown]
# ## Feature extraction

# %%
# Create tokenizer and model
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")
# tokenizer_xlnet = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# Tokenize data
def tokenize_data(tokenizer, texts, max_length=128):
    # Make sure texts is a list of strings
    if not isinstance(texts, list):
        texts = list(texts)
    
    # Check for any non-string entries and convert them
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            texts[i] = str(text)
    
    encodings = tokenizer(
        texts, 
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings['input_ids'], encodings['attention_mask']

# Tokenize train and validation data
X_train_inputs, X_train_masks = tokenize_data(tokenizer_bert, x_train["cleaned_text"].tolist())
X_val_inputs, X_val_masks = tokenize_data(tokenizer_bert, x_val["cleaned_text"].tolist())

# Tf-idf features
vectorizer = TfidfVectorizer()
# x_train_tfidf = vectorizer.fit_transform(x_train["cleaned_text"]).toarray()
# x_val_tfidf = vectorizer.transform(x_val["cleaned_text"]).toarray()

# Numerical features
x_train_num = x_train.drop(["cleaned_text", "selected_text"], axis = 1).values
x_val_num = x_val.drop(["cleaned_text", "selected_text"], axis = 1).values

# Convert labels to one-hot if needed (depends on your loss function)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val_tf = tf.convert_to_tensor(y_val, dtype=tf.int32)

# Create TensorFlow datasets
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_inputs, X_train_masks, x_train_num), y_train_tf))
train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_inputs, X_val_masks, x_val_num), y_val_tf))
val_dataset = val_dataset.batch(batch_size)

# %% [markdown]
# ## Transformer model architecture

# %%
# Define a transformer-based model for sentiment analysis using TensorFlow
class TransformerSentimentClassifier(tf_keras.Model):
    def __init__(self, num_classes=3):
        super(TransformerSentimentClassifier, self).__init__()
        self.bert = TFAutoModel.from_pretrained("bert-base-uncased")
        self.dropout = tf_keras.layers.Dropout(0.1)
        self.layer1 = tf_keras.layers.Dense(128, activation="relu")
        self.layer2 = tf_keras.layers.Dense(64, activation="relu")
        self.layer3 = tf_keras.layers.Dense(64, activation="relu")
        self.layer_num1 = tf_keras.layers.Dense(64, activation="relu")
        self.classifier = tf_keras.layers.Dense(num_classes, activation="softmax")
        
    def call(self, inputs, training=False):
        # Get transformer outputs
        ids, masks, num = inputs
        outputs = self.bert(input_ids=ids, attention_mask=masks)
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Linear Layer for every input
        layer2 = self.layer2(self.layer1(pooled_output))
        layer2_num = self.layer_num1(num)
        concatenated_output = tf_keras.layers.Concatenate()([layer2, layer2_num])
        final_layer = self.layer3(concatenated_output)

        # Apply dropout and the classification layer
        x = self.dropout(final_layer, training=training)
        logits = self.classifier(x)
        
        return logits

model = TransformerSentimentClassifier()

# Compile the model
# Enable mixed precision training
from tf_keras.mixed_precision import Policy, set_global_policy
policy = Policy('mixed_float16')
set_global_policy(policy)

optimizer = tf_keras.optimizers.Adam(learning_rate=2e-5)
loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)

# Training parameters
epochs = 3
steps_per_epoch = len(train_dataset)
total_steps = steps_per_epoch * epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

# Callbacks
lr_scheduler = tf_keras.callbacks.LearningRateScheduler(
    lambda epoch: optimizer.learning_rate * (epoch / warmup_steps) if epoch < warmup_steps else optimizer.learning_rate
)
early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

# Train the model
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[lr_scheduler, early_stopping]
)

# %% [markdown]
# ## Saving and loading the model

# %%
# Save the model after training
# 1. Save the entire model (including optimizer state)
model.save_weights('all_at_once_weights.h5')

# 2. Save the model architecture as JSON (optional)
model_json = model.to_json()
with open("all_at_once.json", "w") as json_file:
    json_file.write(model_json)

print("Model saved successfully")

# Later, to load the model:
def load_trained_model(model_name, num_classes=3):
    # Recreate the model architecture
    loaded_model = TransformerSentimentClassifier(model_name=model_name, num_classes=num_classes)
    
    # Compile the model to initialize weights
    loaded_model.compile(
        optimizer=tf_keras.optimizers.Adam(),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Build the model with a sample input
    sample_input_ids = tf.ones((1, 128), dtype=tf.int32)
    sample_masks = tf.ones((1, 128), dtype=tf.int32)
    _ = loaded_model([sample_input_ids, sample_masks])
    
    # Load the weights
    loaded_model.load_weights('all_at_once_weights.h5')
    return loaded_model

# Example of loading the model
# loaded_model = load_trained_model()

# %% [markdown]
# ## Saving and loading in TensorFlow format

# %%
# Save the entire model in SavedModel format
model.save('all_at_once', save_format='tf')

# Later, to load:
# loaded_model = tf_keras.models.load_model('transformer_sentiment_model_saved')

# %% [markdown]
# ## Load test data

# %%
test = pd.read_csv("data/test_merged.csv").dropna()
y_test = label_encoder.transform(test["sentiment"].values)
x_test = test.drop("sentiment", axis = 1)
X_test_inputs, X_test_masks = tokenize_data(tokenizer_bert, x_test["cleaned_text"].tolist())
# x_test_tfidf = vectorizer.transform(x_test["cleaned_text"])
x_test_num = x_test.drop(["cleaned_text"], axis = 1).values

# %% [markdown]
# ## Evaluate models

# %%
reverse_label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Make predictions
predictions = model.predict((X_test_inputs, X_test_masks, x_test_num))
predictions = np.argmax(predictions, axis=1)
mapped_predictions = np.array([reverse_label_map[prediction] for prediction in predictions])

test["predicted_sentiment"] = mapped_predictions
test.to_csv("data/test_predictions.csv", index=False)

test.head()

# %%
print(classification_report(test["sentiment"], test["predicted_sentiment"]))


