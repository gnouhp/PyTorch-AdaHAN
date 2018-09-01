import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import time
from models import AdaHAN, AttentionVisualizer
from vqa_generator import vqa_data



device = 'cpu'
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
starttime = time.time()

# Make and save the images, questions, and answers.
n_samples = 500
train_samples = 400
vqa_data(n_samples)

# Load the questions and answers.
def load_data(path2questions, path2answers):
    n_words = 0
    n_answers = 0
    word2idx = {}
    idx2answer = []
    answer2idx = {}
    questions = []
    answers = []

    with open(path2questions, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            split_qs = [question.split(' ') for question in line.split(',')[:-1]]
            questions.append(split_qs)
            for q in split_qs:
                for word in q:
                    if word not in word2idx:
                        word2idx[word] = n_words
                        n_words +=1
        
    with open(path2answers, 'r') as f:
        for line in f:
            line = line.rstrip()
            split_as = line.split(',')[:-1]
            answers.append(split_as)
            for word in split_as:
                if word not in answer2idx:
                    answer2idx[word] = n_answers
                    idx2answer.append(word)
                    n_answers +=1

    return n_words, n_answers, word2idx, idx2answer, answer2idx, questions, answers

path2questions = "data/questions.txt"
path2answers = "data/answers.txt"
n_words, n_answers, word2idx, idx2answer, answer2idx, questions, answers = load_data(path2questions, path2answers)


def sentence2tensor(question):
    ''' Takes a sentence and outputs a torch LongTensor, which is accepted by AdaHAN's nn.Embedding module. '''
    return torch.tensor([word2idx[word] for word in question]).long()


vocab_size = n_words
n_classes = n_answers
n_epochs = 2000
model = AdaHAN(vocab_size, hidden_size=8, n_classes=n_classes, k=2, adaptive=False)  # When we set adaptive to false, we're actually using the HAN algorithm.
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

sliding_accuracy = deque(maxlen=300)  # to help track and log the training accuracy
sliding_vaccuracy = deque(maxlen=300)  # for the validation accuracy

n_correct = 0
attn_vis = AttentionVisualizer()
plotting_timestep = []
plotting_accuracy = []
plotting_vaccuracy = []  # validation accuracy

plt.ion()  # set plt to interactive mode


def sample(train_sample=True):
    ''' A function to get the relevant image, question, and answers for a sample from the training or validation set. '''
    if train_sample:
        sample_idx = randint(train_samples)
    else:
        sample_idx = randint(low=train_samples, high=n_samples)

    image_pil = Image.open("data/images/sample_{}.jpg".format(sample_idx))
    image_tensor = ToTensor()(image_pil)
    image_tensor = image_tensor.unsqueeze(dim=0)
    sample_questions = questions[sample_idx]
    sample_answers = answers[sample_idx]

    return image_pil, image_tensor, sample_questions, sample_answers
    
# training loop
for epoch_i in range(n_epochs):
    image_pil, image_tensor, sample_questions, sample_answers = sample(train_sample=True)
    for idx, question in enumerate(sample_questions):
        sentence_tensor = sentence2tensor(question)
        target_class = torch.tensor(answer2idx[sample_answers[idx]]).long().unsqueeze(dim=0)
        class_preds, latent_mask = model(image_tensor=image_tensor, sentence_tensor=sentence_tensor)
        loss = criterion(class_preds, target_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_class = torch.max(class_preds.view(-1), dim=0)[1]

        if pred_class == answer2idx[sample_answers[idx]]:
            sliding_accuracy.append(1)
        else:
            sliding_accuracy.append(0)

    # log and validation visualization step
    if (epoch_i + 1) % 100 == 0:
        with torch.no_grad():
            for val_i in range(50):
                ''' Visualize attention on the input image and question.'''
                image_pil, image_tensor, sample_questions, sample_answers = sample(train_sample=False)
                q_i = randint(3)  # there are 3 questions per image in this dataset.
                question = sample_questions[q_i]
                sentence_tensor = sentence2tensor(question)
                class_preds, latent_mask = model(image_tensor=image_tensor, sentence_tensor=sentence_tensor)
                pred_class = torch.max(class_preds.view(-1), dim=0)[1]

                if pred_class == answer2idx[sample_answers[idx]]:
                    sliding_vaccuracy.append(1)
                else:
                    sliding_vaccuracy.append(0)

        mean_accuracy = np.mean(sliding_accuracy)
        mean_vaccuracy = np.mean(sliding_vaccuracy)
        print("Epoch: {}/{}, Training Accuracy: {:.3f}, Validation Accuracy: {:.3f}".format(epoch_i+1, n_epochs, mean_accuracy, mean_vaccuracy))
        plotting_timestep.append(epoch_i)
        plotting_accuracy.append(mean_accuracy)
        plotting_vaccuracy.append(mean_vaccuracy)
        print("Question: {} Correct Answer: {}. Predicted Answer: {}.".format(" ".join(question), sample_answers[q_i].upper(), idx2answer[pred_class.item()].upper()))
        attended_tensor = attn_vis(image_tensor, latent_mask)
        attended_np = attended_tensor.squeeze(dim=0).permute(1, 2, 0).detach().numpy()
        image_np = image_tensor.squeeze(dim=0).permute(1, 2, 0).detach().numpy()
        fig, (ax_orig, ax_attn) = plt.subplots(ncols=2)
        ax_orig.imshow(image_np)
        ax_attn.imshow(attended_np)
        plt.suptitle(" ".join(question))
        plt.tight_layout()


# plot the training and validation accuracy histories
plt.figure(123)
plt.plot(plotting_timestep, plotting_accuracy, lw=3, label="HAN - Training")
plt.plot(plotting_timestep, plotting_vaccuracy, lw=3, label="HAN - Validation")
plt.plot(plotting_timestep, np.ones_like(plotting_accuracy) * .25, lw=3, label="Random Guessing")
plt.legend()
plt.ylim(ymin=0, ymax=1)
plt.title("VQA Accuracy")
plt.show()

print("Script executed in {:.2f} minutes.".format((time.time() - starttime) / 60 ))
