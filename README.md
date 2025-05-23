# Chatbot with Intent Recognition using PyTorch

This project implements a conversational chatbot that uses natural language processing techniques and a custom neural network to classify user input into predefined intents. It is built using Python, PyTorch, and NLTK.

## Overview

The chatbot preprocesses input using tokenization and stemming, then converts each sentence into a bag-of-words representation. These vectors are fed into a feedforward neural network trained to recognize different user intents based on a JSON file (`intents.json`) that defines all supported categories and responses.

Once trained, the model is saved and reused without retraining, enabling fast interaction with users. During conversation, the model predicts the most likely intent and responds accordingly, choosing randomly from the set of predefined responses for that intent.

## Key Features

- Intent classification using a custom PyTorch neural network.
- Preprocessing with NLTK: tokenization, stemming, and bag-of-words.
- Structured training data defined in a JSON file for easy customization.
- Chat interface via command-line with probabilistic response handling.
- Model persistence for efficient reuse.

## Additional Notes

The chatbot was designed for educational and experimental purposes, making it easy to modify and expand. By adjusting the JSON file or the training parameters, developers can adapt it to more complex domains or extend it into a more sophisticated AI assistant.

---
