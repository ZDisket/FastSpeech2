import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocessor.emotion import EmotionProcessor
from model.zephyr import Zephyr


class ZephyrFrontEnd:
    def __init__(self, model_path, processor, device='cpu'):
        """
        Initializes the ZephyrFrontEnd with a given model path and processor.

        Args:
            model_path (str): Path to the model checkpoint.
            processor (EmotionProcessor): An instance of EmotionProcessor for text processing.
            device (str, optional): The device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device = device
        self.processor = processor
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the model from the given path.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: The loaded model.
        """
        model = Zephyr(len(self.processor.char_to_index) + 1,
                       13, n_heads=4, num_conv_layers=4, dropout=0.1,
                       hidden_dim=256, kernel_sizes=[3, 3, 4, 5])
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model.eval()
        return model

    def predict_emotions(self, text):
        """
        Predicts the emotion probabilities for a given text.

        Args:
            text (str): The input text string.

        Returns:
            np.ndarray: The predicted probabilities for each emotion.
            tuple: tensors: (detached hidden states size (batch, n_blocks, seq_len, hidden_dim), final hidden states (batch, 1, hidden_dim))
        """
        indices = self.processor.text_to_sequence(text)
        indices_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)  # Add batch dimension
        lengths_tensor = torch.LongTensor([len(indices)]).to(self.device)

        with torch.no_grad():
            logits, hid_tup = self.model(indices_tensor, lengths_tensor)

        probabilities = torch.softmax(logits, dim=1).squeeze(
            0).cpu().numpy()  # Remove batch dimension and convert to numpy
        return probabilities, hid_tup

    def plot_emotion_probabilities(self, text):
        """
        Given a text, processes it through the model and plots the probabilities for each emotion.

        Args:
            text (str): The input text string.
        """
        probabilities, _ = self.predict_emotions(text)
        emotions = self.processor.emotion_list

        plt.figure(figsize=(8, 4))
        plt.bar(emotions, probabilities, color='skyblue')
        plt.xlabel('Emotions')
        plt.ylabel('Probability')
        plt.title('Emotion Probabilities')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.show()