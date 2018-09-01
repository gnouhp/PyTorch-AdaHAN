import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = 'cpu'

class AttentionVisualizer(nn.Module):
    ''' A class that accepts a latent mask, and uses a dummy input variable
        passed through convolutional layers of identical kernel size,
        padding, and stride as those of the EncoderCNN in HAN and AdaHAN.
        By backpropagating the latent output to the dummy input, we can pinpoint
        the locations in the original input image that correspond to spatial locations in the latent,
        which allows us to visualize binary attention masks.
    '''
    def __init__(self):
        super(AttentionVisualizer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=1)
        
        self.dummy_input = torch.ones(1, 3, 256, 256, requires_grad=True)


    def forward(self, input_image, latent_mask):
        self.zero_grad()  # reset the gradients
        x = self.pool(self.conv1(self.dummy_input))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = torch.sum(x.view(-1) * latent_mask)
        x.backward()
        ''' the following lines are helpful print statements for seeing what percentage of spatial locations are selected. '''
        print("Number of latent spatial locations selected for attention: {}.".format(torch.sum(latent_mask).item()))
        n_selected_pixels = torch.sum(torch.where(self.dummy_input.grad != 0, torch.ones_like(input_image), torch.zeros_like(input_image))).item()
        print("Percentage of input image pixels attended to: {:.1f}%.\n".format(n_selected_pixels/(3 * 256 * 256) * 100))  # in this case, our images are 3x256x256 pixels

        attended_image = torch.where(self.dummy_input.grad != 0, input_image, torch.ones_like(input_image))  # this whites out the unattended spatial locations
        return attended_image



class EncoderCNN(nn.Module):

    def __init__(self, hidden_size):
        super(EncoderCNN, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))  # 128x128
        x = self.relu(self.pool(self.conv2(x)))  # 64x64
        x = self.relu(self.pool(self.conv3(x)))  # 32x32
        return x



class EncoderRNN(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, sentence_tensor):
        hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        for word in sentence_tensor:
            embedded_input = self.embedding(word).view(1, 1, -1)
            output, hidden = self.gru(embedded_input, hidden)
        return hidden



class AdaHAN(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, n_classes, k=2, adaptive=False):
        # k is the number of spatial locations attended to in the latent representation.
        super(AdaHAN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.encoder_cnn = EncoderCNN(hidden_size)
        self.encoder_rnn = EncoderRNN(vocab_size, hidden_size)
        self.k = k
        self.adaptive = adaptive
        # 1x1 conv for lowering the number of channels from the hidden size to 2.
        self.conv1by1 = nn.Conv2d(hidden_size, 2, kernel_size=1, padding=0, stride=1)  
        self.fc1 = nn.Linear(32 ** 2, n_classes)


    def forward(self, image_tensor, sentence_tensor):
        encoded_img = self.encoder_cnn(image_tensor)
        encoded_sentence = self.encoder_rnn(sentence_tensor).view(self.hidden_size, 1, 1)

        encoded_sentence = encoded_sentence.repeat((1, 32, 32))  # Nx8x32x32 is the shape of encoded_img
        encoded_sum = encoded_img + encoded_sentence
        representation = self.conv1by1(encoded_sum).squeeze(dim=0)
        presence_vector = torch.sum(representation ** 2, dim=0).view(-1)  # The presence vector is the L2 norm of the embedding at each spatial location.
        topk_idxes = torch.topk(presence_vector, k=self.k)[1]
        m_vector = torch.sum(representation, dim=0).view(-1)  # The 'm' vector is a element-wise sum of the multimodal values of the spatial embedding, reshaped into a vector.

        # We'll need the latent mask to later visualize the hard attention over the input image.
        ''' select spatial locations based on AdaHAN's softmax threshold or HAN's topk l2 activations. '''
        if self.adaptive:
            # The 1/(len(encoded_vector)) threshold is a design choice based on the uniform probability distribution over each of the latent spatial locations.
            latent_mask = torch.where(F.softmax(presence_vector, dim=0) >= (1/len(m_vector)), torch.ones_like(presence_vector), torch.zeros_like(presence_vector))  
        else:
            latent_mask = torch.zeros_like(presence_vector)
            latent_mask[topk_idxes] = 1
        attended_vector = m_vector * latent_mask  # zeros out the activations at masked spatial locations
        class_preds = F.log_softmax(self.fc1(attended_vector).unsqueeze(dim=0), dim=1)
        return class_preds, latent_mask
