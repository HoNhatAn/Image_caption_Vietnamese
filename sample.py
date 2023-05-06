import torch
import pickle
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from PIL import Image
from build_vocab import *

def load_image(image_path, transform=None):
    image = image_path.convert('RGB')
    image = image.resize([256, 256], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def run_inference(image_path, encoder_path, decoder_path, vocab_path, embed_size=256, hidden_size=512, num_layers=1):
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])



    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption).replace('<start>', '')
    sentence = sentence.replace('<end>', '')
    sentence = sentence.replace('_', ' ')

    # Print out the image and the generated caption
    return sentence.strip().capitalize()
def runn(image):
    encoder_path = "encoder-50-20.ckpt"
    decoder_path = "decoder-50-20.ckpt"
    vocab_path = "preprocessed_vocab_.pkl"
    returned_sentence = run_inference(image, encoder_path, decoder_path, vocab_path)
    return returned_sentence
if __name__ == '__main__':
    image=Image.open("./AnhDaLuu.jpg")
    print(runn(image))