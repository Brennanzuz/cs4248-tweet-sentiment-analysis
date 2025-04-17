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
import pickle

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from keras.models import load_model

# Load pre-trained embeddings
import gensim.downloader as api
glove_vectors = api.load("glove-twitter-200")
embedding_dim = glove_vectors.vector_size

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
# ## Load and split data

# %%
train_df = pd.read_csv("data/train_preprocessed.csv", encoding="ISO-8859-1")
train_set, validation_set = train_test_split(train_df, test_size=0.2, random_state=20250310)

print("Train set size: " + str(len(train_set)))
train_set.head()

# %%
print("Validation set size: " + str(len(validation_set)))
validation_set.head()

# %% [markdown]
# ## Feature extraction

# %%
# Tokenize data and prepare for model input
def tokenize_data(texts, tokenizer=None, max_length=128):
    # Create or use tokenizer
    if tokenizer is None:
        tokenizer = tf_keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = tf_keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post'
    )
    
    return padded_sequences, tokenizer

# Create embedding matrix from GloVe
def create_embedding_matrix(tokenizer, glove_vectors, embedding_dim=200):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in tokenizer.word_index.items():
        if word in glove_vectors:
            embedding_matrix[idx] = glove_vectors[word]
    
    return embedding_matrix

# Process the data
# Hotfix because for some reason the preprocessing didn't ALREADY replace the NaN values
train_set["lemmatized_sentence"] = train_set["lemmatized_sentence"].fillna("").astype(str)
validation_set["lemmatized_sentence"] = validation_set["lemmatized_sentence"].fillna("").astype(str)

label_encoder = LabelEncoder()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_set["lemmatized_sentence"].tolist())

# Prepare input sequences
X_train_seq, _ = tokenize_data(train_set["lemmatized_sentence"].tolist(), tokenizer)
X_val_seq, _ = tokenize_data(validation_set["lemmatized_sentence"].tolist(), tokenizer)
y_train = label_encoder.fit_transform(train_set["sentiment"])
y_val = label_encoder.transform(validation_set["sentiment"])

# Create embedding matrix from GloVe
embedding_matrix = create_embedding_matrix(tokenizer, glove_vectors)

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train))
train_dataset = train_dataset.shuffle(len(X_train_seq)).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val))
val_dataset = val_dataset.batch(16)

# %% [markdown]
# ## Transformer model architecture

# %%
# Static GloVe Classifier - No contextual processing
class StaticGloVeClassifier(tf_keras.Model):
    def __init__(self, embedding_matrix, max_length=128, num_classes=3, dropout_rate=0.1):
        super(StaticGloVeClassifier, self).__init__()
        
        vocab_size = embedding_matrix.shape[0]
        d_model = embedding_matrix.shape[1]  # Should be 200 for glove-twitter-200
        
        # GloVe embedding layer (initialized with pretrained weights)
        self.embedding = tf_keras.layers.Embedding(
            vocab_size, d_model,
            weights=[embedding_matrix],
            trainable=False  # Keep truly static - no fine-tuning
        )
        
        # Simple feed-forward classifier on top of static embeddings
        # Approach 1: Global Average Pooling (average word vectors)
        self.pooling = tf_keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf_keras.layers.Dense(128, activation='relu')
        self.dropout = tf_keras.layers.Dropout(dropout_rate)
        self.classifier = tf_keras.layers.Dense(num_classes)
        
    def call(self, inputs, training=False):
        # Get embeddings
        x = self.embedding(inputs)  # (batch_size, seq_len, d_model)
        
        # Global average pooling (average of all word vectors in tweet)
        x = self.pooling(x)
        
        # Classification head
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.classifier(x)
        
        return output
    
# Create static GloVe model
model = StaticGloVeClassifier(
    embedding_matrix=embedding_matrix,
    max_length=128,
    num_classes=3
)

# Compile the model
# Enable mixed precision training
from tf_keras.mixed_precision import Policy, set_global_policy
policy = Policy('mixed_float16')
set_global_policy(policy)

# Compile model
optimizer = tf_keras.optimizers.Adam(learning_rate=1e-4)
loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Define a learning rate scheduler (optional)
class WarmupScheduler(tf_keras.callbacks.Callback):
    def __init__(self, warmup_steps, total_steps, initial_lr=2e-5, min_lr=0):
        super(WarmupScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.global_step = 0
        
    def on_batch_begin(self, batch, logs=None):
        self.global_step += 1
        if self.global_step < self.warmup_steps:
            lr = self.global_step / self.warmup_steps * self.initial_lr
        else:
            decay_steps = self.total_steps - self.warmup_steps
            decay_rate = (self.min_lr - self.initial_lr) / decay_steps
            lr = self.initial_lr + decay_rate * (self.global_step - self.warmup_steps)
            lr = max(lr, self.min_lr)
        
        tf_keras.backend.set_value(self.model.optimizer.lr, lr)

# Train the model
epochs = 3
early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# Alternatively, if you prefer a simpler approach without the custom learning rate scheduler:
# history = model.fit(
#     train_dataset,
#     epochs=epochs,
#     validation_data=val_dataset
# )

# %% [markdown]
# ## Saving and loading the model

# %%
# Save the model after training
# 1. Save the entire model (including optimizer state)
model.save_weights('glove_transformer_sentiment_model_weights.h5')

# 2. Save the tokenizer vocabulary (critical for recreating embedding matrix)
with open('glove_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Save hyperparameters (optional but helpful)
model_params = {
    'max_length': 128,
    'num_classes': 3,
    'num_layers': 4,
    'd_model': 200,
    'num_heads': 8,
    'dff': 512,
    'dropout_rate': 0.1
}

with open('glove_model_params.pickle', 'wb') as handle:
    pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model saved successfully")

# Later, to load the model:
def load_trained_glove_model(glove_vectors_path="glove-twitter-200"):
    # 1. Load the tokenizer
    with open('glove_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # 2. Load model parameters
    with open('glove_model_params.pickle', 'rb') as handle:
        model_params = pickle.load(handle)
    
    # 3. Load GloVe vectors
    import gensim.downloader as api
    glove_vectors = api.load(glove_vectors_path)
    
    # 4. Recreate the embedding matrix
    embedding_matrix = create_embedding_matrix(tokenizer, glove_vectors, embedding_dim=model_params['d_model'])
    
    # 5. Recreate the model architecture
    loaded_model = GloVeTransformerClassifier(
        embedding_matrix=embedding_matrix,
        max_length=model_params['max_length'],
        num_classes=model_params['num_classes'],
        num_layers=model_params['num_layers'],
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        dff=model_params.get('dff', 512),
        dropout_rate=model_params.get('dropout_rate', 0.1)
    )
    
    # 6. Compile the model to initialize weights
    loaded_model.compile(
        optimizer=tf_keras.optimizers.Adam(),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # 7. Build the model with a sample input (single tensor, not tuple like BERT)
    sample_input = tf.ones((1, model_params['max_length']), dtype=tf.int32)
    _ = loaded_model(sample_input)
    
    # 8. Load the weights
    loaded_model.load_weights('glove_transformer_sentiment_model_weights.h5')
    
    return loaded_model, tokenizer

# Example of loading and using the model:
# loaded_model, loaded_tokenizer = load_trained_glove_model()
# 
# # Process new text
# def predict_sentiment(text, model, tokenizer, max_length=128):
#     # Preprocess text (similar to your training preprocessing)
#     # ... your preprocessing steps here ...
#     
#     # Tokenize
#     sequences = tokenizer.texts_to_sequences([text])
#     padded = tf_keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)
#     
#     # Predict
#     logits = model(padded)
#     probabilities = tf.nn.softmax(logits, axis=-1)
#     predicted_class = tf.argmax(probabilities, axis=-1)
#     
#     # Map class index to label (depends on your label encoding)
#     labels = ['negative', 'neutral', 'positive']  # Adjust based on your encoding
#     return labels[predicted_class[0]]

# %% [markdown]
# ## Saving and loading in TensorFlow format

# %%
# Save the entire model in SavedModel format
model.save('transformer_sentiment_model_saved', save_format='tf')

# Later, to load:
loaded_model = tf_keras.models.load_model("transformer_sentiment_model_saved")
with open('glove_tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# %% [markdown]
# ## Load test data

# %%
test_df = pd.read_csv("data/test_preprocessed.csv")
print("Test set size: " + str(len(test_df)))

X_test_seq, _ = tokenize_data(test_df["lemmatized_sentence"].tolist(), tokenizer)

# %% [markdown]
# ## Evaluate models

# %%
reverse_label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Make predictions
predictions = model.predict(X_test_seq)
predictions = np.argmax(predictions, axis=1)
mapped_predictions = np.array([reverse_label_map[prediction] for prediction in predictions])

test_df["predicted_sentiment"] = mapped_predictions
test_df.to_csv("data/test_predictions_glove.csv", index=False)

test_df.head()

# %%
# Show sample predictions
print("\nSample predictions:")
sample_results = test_df[["lemmatized_sentence", "predicted_sentiment"]].head(10)
for i, row in sample_results.iterrows():
    print(f"Text: {row['lemmatized_sentence'][:50]}{'...' if len(row['lemmatized_sentence']) > 50 else ''}")
    print(f"Predicted sentiment: {row['predicted_sentiment']}\n")

# Calculate class distribution in predictions
sentiment_counts = test_df["predicted_sentiment"].value_counts()
print("\nPredicted sentiment distribution:")
print(sentiment_counts)
print(f"Negative: {sentiment_counts.get('negative', 0) / len(test_df):.2%}")
print(f"Neutral: {sentiment_counts.get('neutral', 0) / len(test_df):.2%}")
print(f"Positive: {sentiment_counts.get('positive', 0) / len(test_df):.2%}")
print(classification_report(test_df["sentiment"], test_df["predicted_sentiment"]))


