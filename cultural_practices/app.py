import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

vectorizer = joblib.load("cultural_practices/vectorizer.pkl")
festival_encoder = joblib.load("cultural_practices/festival_encoder.pkl")
practice_encoder = joblib.load("cultural_practices/practice_encoder.pkl")

input_size = 62  
hidden_size = 128
num_layers = 1
num_festivals = len(festival_encoder.classes_)
num_practices = len(practice_encoder.classes_)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_festivals, num_practices):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_festival = nn.Linear(hidden_size, num_festivals)
        self.fc_practice = nn.Linear(hidden_size, num_practices)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = out[:, -1, :]
        festival_output = self.fc_festival(out)
        practice_output = self.fc_practice(out)
        return festival_output,practice_output

model = LSTMClassifier(input_size, hidden_size, num_layers, num_festivals, num_practices)
model.load_state_dict(torch.load("cultural_practices/lstm_model.pth"))
model.eval()

def predict_festival_and_practice(transcript):
    model.eval()
    with torch.no_grad():
        text_tfidf = vectorizer.transform([transcript]).toarray()
        text_tensor = torch.tensor(text_tfidf, dtype=torch.float32)

        festival_output, practice_output = model(text_tensor)
        festival_id = torch.argmax(festival_output, dim=1).item()
        practice_id = torch.argmax(practice_output, dim=1).item()

        predicted_festival = festival_encoder.inverse_transform([festival_id])[0]
        predicted_practice = practice_encoder.inverse_transform([practice_id])[0]

    return predicted_festival, predicted_practice

example_text = "During Holi, we follow the tradition of cleaning water bodies before festival."
pred_festival, pred_practice = predict_festival_and_practice(example_text)

print("Predicted Festival:",pred_festival)
print("Cultural Practice:",pred_practice)