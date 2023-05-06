from fastapi import File, UploadFile, FastAPI
import sample
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import os

from build_vocab import Vocabulary

cwd = os.getcwd()
app = FastAPI()


class FileUpload(object):
    def __init__(self):

        self.encoder_path = "encoder-50-20.ckpt"
        self.decoder_path = "decoder-50-20.ckpt"
        self.vocab_path = "preprocessed_vocab_.pkl"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def result(self, image_path, embed_size=256, hidden_size=512, num_layers=1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        with open(self.vocab_path, 'rb') as f:
            print("using " + self.vocab_path)
            self.vocab = pickle.load(f)
        encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(embed_size, hidden_size, len(self.vocab), num_layers)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)

        encoder.load_state_dict(torch.load(self.encoder_path, map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(self.decoder_path, map_location=torch.device('cpu')))

        # Prepare an image

        image = image_path.resize([256, 256], Image.LANCZOS)
        image = transform(image).unsqueeze(0)
        image_tensor = image.to(self.device)

        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            print(word)
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption).replace('<start>', '')
        sentence = sentence.replace('<end>', '')
        sentence = sentence.replace('_', ' ')

        print(sentence)
        return sentence.strip().capitalize()


@app.post('/books')
async def list_books(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    img = Image.open(file.filename).convert('RGB')
    help = FileUpload()
    result = help.result(image_path=img)

    return {'data': result}