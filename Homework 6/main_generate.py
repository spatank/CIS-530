pip install unidecode

import unidecode
import string
import random
import re
from models import *
import time, math

all_characters = string.printable
n_characters = len(all_characters)

# Load data 
root_directory = '../data'
text_directory = 'jane_austen.txt'
file = open(root_directory + text_directory, encoding = 'UTF-8', errors = 'ignore').read()
file_len = len(file)

chunk_len = 200

def random_chunk():
  start_index = random.randint(0, file_len - chunk_len)
  end_index = start_index + chunk_len + 1
  return file[start_index:end_index]

# Turn string into list of longs
def char_tensor(string):
  tensor = torch.zeros(len(string)).long()
  for c in range(len(string)):
    tensor[c] = all_characters.index(string[c])
  return Variable(tensor)

def random_training_set():    
  chunk = random_chunk()
  inp = char_tensor(chunk[:-1])
  target = char_tensor(chunk[1:])
  return inp, target

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
  hidden = decoder.init_hidden()
  prime_input = char_tensor(prime_str)
  predicted = prime_str
  
  # Use priming string to "build up" hidden state
  for p in range(len(prime_str) - 1):
    _, hidden = decoder(prime_input[p], hidden)
  inp = prime_input[-1]
  
  for p in range(predict_len):
    output, hidden = decoder(inp, hidden)
        
    # Sample from the network as a multinomial distribution
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1)[0]
    
    # Add predicted character to string and use as next input
    predicted_char = all_characters[top_i]
    predicted += predicted_char
    inp = char_tensor(predicted_char)

  return predicted

def time_since(since):
  s = time.time() - since
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def train(inp, target):
  hidden = decoder.init_hidden()
  decoder.zero_grad()
  loss = 0

  for c in range(chunk_len):
      output, hidden = decoder(inp[c], hidden)
      loss += criterion(output, target[[c]])

  loss.backward()
  decoder_optimizer.step()
  return loss.data / chunk_len

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

decoder = RNN(n_characters, hidden_size, n_characters, n_layers) # RNN imported from models.py
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
  loss = train(*random_training_set())       
  loss_avg += loss

  if epoch % print_every == 0:
      print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
      print(evaluate('Wh', 100), '\n')

  if epoch % plot_every == 0:
      all_losses.append(loss_avg / plot_every)
      loss_avg = 0

