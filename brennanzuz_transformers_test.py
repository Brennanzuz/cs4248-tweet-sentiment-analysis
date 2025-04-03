# %% [markdown]
# # Transformer model

# %%
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, TFAutoModel 
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
# ## Feature extraction

# %%
# Create tokenizer and model
model_name = "bert-base-uncased"  # You can use other models like "roberta-base" or "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
label_encoder = LabelEncoder()

# Tokenize data
def tokenize_data(texts, max_length=128):
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

# %% [markdown]
# ## Transformer model architecture

# %%
# Define a transformer-based model for sentiment analysis using TensorFlow
class TransformerSentimentClassifier(tf_keras.Model):
    def __init__(self, model_name="bert-base-uncased", num_classes=3):
        super(TransformerSentimentClassifier, self).__init__()
        self.transformer = TFAutoModel.from_pretrained(model_name)
        self.dropout = tf_keras.layers.Dropout(0.1)
        self.classifier = tf_keras.layers.Dense(num_classes, activation=None)
        
    def call(self, inputs, training=False):
        # Get transformer outputs
        input_ids, attention_mask = inputs
        outputs = self.transformer.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and the classification layer
        x = self.dropout(pooled_output, training=training)
        logits = self.classifier(x)
        
        return logits

# Later, to load the model:
def load_trained_model(model_name="bert-base-uncased", num_classes=3):
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
    loaded_model.load_weights('transformer_sentiment_model_weights.h5')
    return loaded_model

# Example of loading the model
# model = load_trained_model()

# Later, to load:
model = tf_keras.models.load_model('transformer_sentiment_model_saved')

# %% [markdown]
# ## Load test data

# %%
test_df = pd.read_csv("data/test_preprocessed.csv")
print("Test set size: " + str(len(test_df)))

X_test_inputs, X_test_masks = tokenize_data(test_df["lemmatized_sentence"].tolist())

# %% [markdown]
# ## Evaluate models

# %%
reverse_label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Make predictions
predictions = model.predict((X_test_inputs, X_test_masks))
predictions = np.argmax(predictions, axis=1)
mapped_predictions = np.array([reverse_label_map[prediction] for prediction in predictions])

test_df["predicted_sentiment"] = mapped_predictions
test_df.to_csv("data/tree_model_predictions.csv", index=False)

test_df.head()

# %%
print(classification_report(test_df["sentiment"], test_df["predicted_sentiment"]))


