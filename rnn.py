import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, word_vector, hidden_state):
        hidden_state = self.rnn_cell(word_vector, hidden_state)
        output = self.fc(hidden_state)
        return output, hidden_state

# Initialize vocabulary
custom_vocabulary = ["hello", "world", "this", "is", "a", "test", "RNN", "model", "prediction", "example"]
vocab_size = len(custom_vocabulary)
embedding_dim = 10  # Size of word embeddings

# Map words to indices and vice versa
word_to_index = {word: idx for idx, word in enumerate(custom_vocabulary)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Create an embedding matrix
embedding_matrix = nn.Embedding(vocab_size, embedding_dim)

# Initialize the RNN model
hidden_size = 4
my_rnn = SimpleRNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=vocab_size)
hidden_state = torch.zeros(1, hidden_size)  # Initial hidden state

# Function to predict the next word
def predict_next_word(sentence, vocabulary):
    global hidden_state
    for word in sentence:
        if word not in vocabulary:
            raise ValueError(f"Word '{word}' is not in the provided vocabulary.")
        word_index = word_to_index[word]
        word_vector = embedding_matrix(torch.tensor([word_index]))
        prediction, hidden_state = my_rnn(word_vector, hidden_state)

    probabilities = F.softmax(prediction, dim=1)
    most_likely_index = torch.argmax(probabilities, dim=1).item()
    return index_to_word[most_likely_index]

# Streamlit UI
st.title("RNN Next Word Prediction")

# Input for vocabulary
custom_vocab_input = st.text_area("Define your vocabulary (comma-separated):", 
                                  value="hello,world,this,is,a,test,RNN,model,prediction,example")
if custom_vocab_input:
    custom_vocabulary = [word.strip() for word in custom_vocab_input.split(",")]
    vocab_size = len(custom_vocabulary)
    word_to_index = {word: idx for idx, word in enumerate(custom_vocabulary)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    embedding_matrix = nn.Embedding(vocab_size, embedding_dim)

# Input for sentence
sentence_input = st.text_input("Enter a sentence (space-separated):", value="hello this is")
sentence = sentence_input.split()

# Predict next word
if st.button("Predict Next Word"):
    try:
        next_word = predict_next_word(sentence, custom_vocabulary)
        st.success(f"Next word prediction: {next_word}")
    except ValueError as e:
        st.error(str(e))
