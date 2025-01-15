import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gensim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Embedding, Input



# ------------------------ Dataset Paths ------------------------
train_emoticon_path = "./datasets/train/train_emoticon.csv"
valid_emoticon_path = "./datasets/valid/valid_emoticon.csv"
test_emoticon_path = "./datasets/test/test_emoticon.csv"


train_feature_path = "./datasets/train/train_feature.npz"
valid_feature_path = "./datasets/valid/valid_feature.npz"
test_feature_path = "./datasets/test/test_feature.npz"

train_text_seq_path = "./datasets/train/train_text_seq.csv"
valid_text_seq_path = "./datasets/valid/valid_text_seq.csv"
test_text_seq_path = "./datasets/test/test_text_seq.csv"

emoji2vec_path = "emoji2vec.bin"



# -------------------------- Dataset 1: Emoticon LSTM Model --------------------------

print("# -------------------------- Dataset 1--------------------------")
# Load the dataset
train_df = pd.read_csv(train_emoticon_path)
val_df = pd.read_csv(valid_emoticon_path)
test_df = pd.read_csv(test_emoticon_path)

# Split emoji strings into individual emojis
def split_emojis(emoji_string):
    return list(emoji_string)

train_df['input_emoticon'] = train_df['input_emoticon'].apply(split_emojis)
val_df['input_emoticon'] = val_df['input_emoticon'].apply(split_emojis)
test_df['input_emoticon'] = test_df['input_emoticon'].apply(split_emojis)

# Extract labels
y_train = train_df['label'].values
y_val = val_df['label'].values

# Tokenize emojis
tokenizer = Tokenizer(char_level=True, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['input_emoticon'])  # Fit tokenizer on training data
word_index = tokenizer.word_index

# Convert emojis to sequences of integers
X_train = tokenizer.texts_to_sequences(train_df['input_emoticon'])
X_val = tokenizer.texts_to_sequences(val_df['input_emoticon'])
X_test = tokenizer.texts_to_sequences(test_df['input_emoticon'])

# Padding sequences to ensure uniform input length (13 emojis per input)
X_train = pad_sequences(X_train, maxlen=13, padding='post')
X_val = pad_sequences(X_val, maxlen=13, padding='post')
X_test = pad_sequences(X_test, maxlen=13, padding='post')

vocab_size = len(word_index) + 1

print(f"Train Shape: {X_train.shape}")
print(f"Validation Shape: {X_val.shape}")
print(f"Test Shape: {X_test.shape}")


# LSTM model definition
def create_lstm_model():
    model = Sequential()
    model.add(Input(shape=(13,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=32))
    model.add(LSTM(12, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train with 100% of the data
model = create_lstm_model()
model.summary()
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Evaluate on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Dataset 1 LSTM Validation Accuracy : {val_accuracy:.4f}')

# Make predictions on the test set and save to txt without any header
y_test_pred = model.predict(X_test)
y_test_pred_binary = (y_test_pred >= 0.5).astype(int)
pd.DataFrame(y_test_pred_binary).to_csv('pred_emoticon.txt', index=False, header=False)


# ------------------------ Dataset 2: PCA then Classification Models ------------------------
print("# -------------------------- Dataset 2--------------------------")
# Load the dataset
data = np.load(train_feature_path, allow_pickle=True)
valid_data = np.load(valid_feature_path, allow_pickle=True)
test_data = np.load(test_feature_path, allow_pickle=True)

train_deep_X = data['features']
train_deep_Y = data['label']

valid_deep_X = valid_data['features']
valid_deep_Y = valid_data['label']

test_deep_X = test_data['features']

# Flatten the data for PCA
train_X_deep_flattened = train_deep_X.reshape(train_deep_X.shape[0], -1)
valid_X_deep_flattened = valid_deep_X.reshape(valid_deep_X.shape[0], -1)
test_X_deep_flattened = test_deep_X.reshape(test_deep_X.shape[0], -1)

# Perform PCA
n_components = 500
pca = PCA(n_components=n_components)
train_embeddings_pca = pca.fit_transform(train_X_deep_flattened)
validation_embeddings_pca = pca.transform(valid_X_deep_flattened)
test_embeddings_pca = pca.transform(test_X_deep_flattened)

print(f"Train Shape: {train_embeddings_pca.shape}")
print(f"Validation Shape: {validation_embeddings_pca.shape}")
print(f"Test Shape: {test_embeddings_pca.shape}")


# Train models and evaluate
models = {
#     "Ridge Regression": RidgeClassifier(),
#     "XG Boost": XGBClassifier(),
#     "SVM": SVC(C=0.01, degree=3, gamma='scale', kernel='linear'),
    "Logistic Regression": LogisticRegression(max_iter=500),
}


for model_name, model in models.items():
    model.fit(train_embeddings_pca, train_deep_Y)
    y_pred_valid = model.predict(validation_embeddings_pca)
    validation_accuracy = accuracy_score(valid_deep_Y, y_pred_valid)
    print(f"Dataset 2 {model_name} Validation Accuracy: {validation_accuracy:.4f}")
    y_test_pred = model.predict(test_embeddings_pca)
    pd.DataFrame(y_test_pred).to_csv('pred_deepfeat.txt', index=False, header=False)



# -------------------------- Dataset 3: GRU Model for Text Sequences --------------------------
print("# -------------------------- Dataset 3--------------------------")
# Load the dataset
train_df = pd.read_csv(train_text_seq_path)
validation_df = pd.read_csv(valid_text_seq_path)
test_df = pd.read_csv(test_text_seq_path)


X_train = np.array([[int(char) for char in seq] for seq in train_df['input_str']])
y_train = np.array(train_df['label'])

X_val = np.array([[int(char) for char in seq] for seq in validation_df['input_str']])
y_val = np.array(validation_df['label'])

X_test = np.array([[int(char) for char in seq] for seq in test_df['input_str']])

print(f"Train Shape: {X_train.shape}")
print(f"Validation Shape: {X_val.shape}")
print(f"Test Shape: {X_test.shape}")

# Build GRU model
max_length = 50
vocab_size = 10
embedding_dim = 64
model = Sequential()
model.add(Input(shape=(max_length,)))
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding_layer'))
model.add(GRU(units=28, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val),verbose=1)

# Evaluate on validation data
validation_loss, validation_accuracy = model.evaluate(X_val, y_val)
print(f"Dataset 3 GRU Validation Accuracy: {validation_accuracy:.4f}")

# Make predictions on the test set and save to txt
y_test_pred = model.predict(X_test)
y_test_pred_binary = (y_test_pred >= 0.5).astype(int)
pd.DataFrame(y_test_pred_binary).to_csv('pred_textseq.txt', index=False, header=False)


#-----------------Task-2------------------------------------------------
print("# -------------------------- Combined Dataset --------------------------")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import gensim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding, Input

# ---------- Dataset-2 Deep Features Processing----------
data = np.load(train_feature_path, allow_pickle=True)
valid_data = np.load(valid_feature_path, allow_pickle=True)
test_deep_data = np.load(test_feature_path, allow_pickle=True)

train_deep_X = data['features']
train_deep_Y = data['label']

valid_deep_X = valid_data['features']
valid_deep_Y = valid_data['label']

test_deep_X = test_deep_data['features']

train_X_deep_flattened = train_deep_X.reshape(train_deep_X.shape[0], -1)
valid_X_deep_flattened = valid_deep_X.reshape(valid_deep_X.shape[0], -1)
test_X_deep_flattened = test_deep_X.reshape(test_deep_X.shape[0], -1)

pca = PCA(n_components=500)
train_embeddings_pca = pca.fit_transform(train_X_deep_flattened)
validation_embeddings_pca = pca.transform(valid_X_deep_flattened)
test_embeddings_pca = pca.transform(test_X_deep_flattened)

# ---------- Dataset-1 Emoji Processing-----------
e2v = gensim.models.KeyedVectors.load_word2vec_format(emoji2vec_path, binary=True)

def get_emoji_embedding(emoji):
    try:
        return e2v[emoji]
    except KeyError:
        return np.zeros(300)

def get_aggregated_embedding(emoji_sequence):
    embeddings = [get_emoji_embedding(emoji) for emoji in emoji_sequence]
    return np.array(embeddings)

train_df = pd.read_csv(train_emoticon_path)
validation_df = pd.read_csv(valid_emoticon_path)
test_df = pd.read_csv(test_emoticon_path)

train_sequences = train_df['input_emoticon'].values
validation_sequences = validation_df['input_emoticon'].values
test_sequences = test_df['input_emoticon'].values

train_labels = train_df['label'].values
validation_labels = validation_df['label'].values

train_embeddings = np.array([get_aggregated_embedding(seq) for seq in train_sequences])
validation_embeddings = np.array([get_aggregated_embedding(seq) for seq in validation_sequences])
test_embeddings = np.array([get_aggregated_embedding(seq) for seq in test_sequences])

train_X_emoji_flattened = train_embeddings.reshape(train_embeddings.shape[0], -1)
valid_X_emoji_flattened = validation_embeddings.reshape(validation_embeddings.shape[0], -1)
test_X_emoji_flattened = test_embeddings.reshape(test_embeddings.shape[0], -1)

pca_emoji = PCA(n_components=1500)
train_embeddings_pca_emoji = pca_emoji.fit_transform(train_X_emoji_flattened)
validation_embeddings_pca_emoji = pca_emoji.transform(valid_X_emoji_flattened)
test_embeddings_pca_emoji = pca_emoji.transform(test_X_emoji_flattened)

#---------- Dataset-3: Text Sequence ----------
train_text_df = pd.read_csv(train_text_seq_path)
validation_text_df = pd.read_csv(valid_text_seq_path)
test_text_df = pd.read_csv(test_text_seq_path)

X_train_text = np.array([[int(char) for char in seq] for seq in train_text_df['input_str']])
y_train_text = np.array(train_text_df['label'])

X_val_text = np.array([[int(char) for char in seq] for seq in validation_text_df['input_str']])
y_val_text = np.array(validation_text_df['label'])

X_test_text = np.array([[int(char) for char in seq] for seq in test_text_df['input_str']])

# -----------Combine All Dataset Embeddings ----------
merged_data_train = np.concatenate((train_embeddings_pca, train_embeddings_pca_emoji, X_train_text), axis=1)
merged_data_valid = np.concatenate((validation_embeddings_pca, validation_embeddings_pca_emoji, X_val_text), axis=1)
merged_data_test = np.concatenate((test_embeddings_pca, test_embeddings_pca_emoji, X_test_text), axis=1)

print(f"Combined Train Shape: {merged_data_train.shape}")
print(f"Combined Validation Shape: {merged_data_valid.shape}")
print(f"Combined Test Shape: {merged_data_test.shape}")

# -----------Logistic Regression-------------
model = LogisticRegression(C= 1, penalty= 'l2', solver='saga',max_iter=2000)
model.fit(merged_data_train, train_deep_Y)

y_val_pred = model.predict(merged_data_valid)
y_test_pred = model.predict(merged_data_test)

val_accuracy = accuracy_score(valid_deep_Y, y_val_pred)
print(f"Combined Dataset Validation Accuracy: {val_accuracy}")

pd.DataFrame(y_test_pred).to_csv('pred_combined.txt', index=False, header=False)
