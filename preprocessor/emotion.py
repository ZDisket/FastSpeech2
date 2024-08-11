import pandas as pd
import numpy as np
import string
import re
from text.numbers import normalize_numbers


class EmotionProcessor:
    def __init__(self, filepath=""):
        self.filepath = filepath
        self._punctuation = "!'(),.:;?~!*&\" "
        self._letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.emotion_list = [
            "anger", "boredom", "empty", "enthusiasm", "fun",
            "happiness", "hate", "love", "neutral", "relief",
            "sadness", "surprise", "worry"
        ]
        self.sentiment_list = ["0", "1"]
        self.reserved_vocab = ["|", "/"]
        self.emotion_index = self.emotion_to_index(self.emotion_list)
        self.sentiment_index = self.emotion_to_index(self.sentiment_list)
        self.char_to_index = self.build_char_to_index()
        self._whitespace_re = re.compile(r'\s+')

    def build_char_to_index(self):
        vocabulary = list(self._letters) + list(self._punctuation)

        total_vocab = vocabulary + self.reserved_vocab
        return {char: idx for idx, char in enumerate(total_vocab)}

    def emotion_to_index(self, emotions):
        return {emotion: idx for idx, emotion in enumerate(emotions)}

    def clean_text(self, text):
        # Regex to remove mentions and URLs
        cleaned_text = re.sub(r'@\w+|http[s]?://\S+', '', text)
        cleaned_text = cleaned_text.replace("&amp", "").replace("&quot", "\"")
        cleaned_text = re.sub(self._whitespace_re, ' ', cleaned_text)
        cleaned_text = normalize_numbers(cleaned_text)

        return cleaned_text

    def text_to_sequence(self, text):
        text = self.clean_text(text)
        indices = [self.char_to_index.get(char, self.char_to_index['/']) for char in text if
                   char not in self.reserved_vocab]
        indices.append(self.char_to_index['|'])
        return indices

    def one_hot_encode(self, emotion, indexes):
        one_hot = np.zeros(len(indexes), dtype=int)
        one_hot[indexes[emotion]] = 1
        return one_hot

    def process_data(self):
        # Load the CSV file
        df = pd.read_csv(self.filepath, delimiter=',')

        # Process each row to extract and transform the data
        processed_data = [
            (self.one_hot_encode(row['sentiment'], self.emotion_index), self.text_to_sequence(row['content']))
            for _, row in df.iterrows()
        ]

        return processed_data

    def process_sentiments(self):
        # Load the CSV file
        df = pd.read_csv(self.filepath, delimiter=',')

        # Process each row to extract and transform the data
        processed_data = []

        for _, row in df.iterrows():
            try:
                current_dat = (self.one_hot_encode(str(row['Sentiment']), self.sentiment_index), self.text_to_sequence(row['SentimentText']))
                processed_data.append(current_dat)
            except KeyboardInterrupt:
                break
            except:
                print(f"Error processing {row}")
                continue

        return processed_data

class EmotionProcessorV2:
    def __init__(self, filepath=""):
        self.filepath = filepath
        self._punctuation = "!'(),.:;?~!*&\" "
        self._letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.reserved_vocab = ["|", "/"]
        self.char_to_index = self.build_char_to_index()
        self._whitespace_re = re.compile(r'\s+')

    def build_char_to_index(self):
        vocabulary = list(self._letters) + list(self._punctuation)
        total_vocab = vocabulary + self.reserved_vocab
        return {char: idx for idx, char in enumerate(total_vocab)}

    def clean_text(self, text):
        # Remove mentions and URLs
        cleaned_text = re.sub(r'@\w+|http[s]?://\S+', '', text)
        cleaned_text = cleaned_text.replace("&amp", "").replace("&quot", "\"")
        cleaned_text = re.sub(self._whitespace_re, ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()  # Remove leading and trailing whitespace
        return cleaned_text

    def text_to_sequence(self, text):
        text = self.clean_text(text)
        indices = [self.char_to_index.get(char, self.char_to_index['/']) for char in text if char in self.char_to_index]
        indices.append(self.char_to_index['|'])
        return np.array(indices, dtype=np.int32)

    def is_valid_entry(self, row):
        try:
            float_columns = row[1:].astype(float)
            return isinstance(row[0], str) and len(row) == 11 and sum(float_columns) > 0 and len(row[0]) < 180
        except ValueError:
            return False

    def process_file(self):
        processed_data = []

        # Read the file using pandas
        df = pd.read_csv(self.filepath, delimiter='|', header=None)

        # Iterate over each row in the dataframe
        for _, row in df.iterrows():
            if self.is_valid_entry(row):
                text_sequence = self.text_to_sequence(row[0])
                emotion_values = row[1:].astype(float).to_numpy(dtype=np.float32)
                processed_data.append((text_sequence, emotion_values))

        return processed_data

    def save_to_numpy(self, output_file):
        processed_data = self.process_file()
        np.save(output_file, processed_data)