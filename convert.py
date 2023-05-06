import dill
from build_vocab import Vocabulary
# Khởi tạo đối tượng Vocabulary
vocab = Vocabulary()

# Sử dụng dill để pickle đối tượng Vocabulary và lưu vào file 'vocab.pkl'
with open('preprocessed_vocab_.pkl', 'rb') as f:
    dill.load(vocab, f)

# Sử dụng dill để unpickle đối tượng Vocabulary từ file 'vocab.pkl'
with open('vocab1.pkl', 'wb') as f:
    loaded_vocab = dill.dump(f)