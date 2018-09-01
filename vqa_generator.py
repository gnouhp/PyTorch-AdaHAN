import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.random import randint, randn
import random
import os


class VQAGenerator():
    def __init__(self, animals, animals2imgs, barn_img, hay_img, img_dim=256, n_samples=500):
        self.animals = animals
        self.animals2imgs = animals2imgs
        self.barn = barn_img
        self.hay = hay_img
        self.img_dim = img_dim
        self.sample_idx = 0
        self.n_samples = n_samples

        self.img = Image.new('RGBA', size=(img_dim, img_dim), color=(0, 0, 0, 0))
        self.questions = []
        self.answers = []
        self.barn_pos = 0
        self.hay_pos = 170

        self.path2imgs = "data/images"
        self.path2questions = "data/questions.txt"
        self.path2answers = "data/answers.txt"

    def clear_txt(self):
        open(self.path2questions, 'w').close()
        open(self.path2answers, 'w').close()

    def clear_img(self):
        self.answers = []
        self.questions = []
        self.img = Image.new('RGBA', size=(self.img_dim, self.img_dim), color=(0, 0, 0, 0))

    def make_sample(self):
        self.img.paste(self.barn, (self.barn_pos, 75), mask=self.barn)
        animal_idx = randint(len(self.animals))
        self.img.paste(self.animals2imgs[animal_idx], (self.barn_pos + randint(-30, 60), 150), mask=self.animals2imgs[animal_idx])
        self.questions.append("What animal is near the front of the barn?")
        self.answers.append(self.animals[animal_idx])

        animal_idx = randint(len(self.animals))
        self.img.paste(self.animals2imgs[animal_idx], (self.barn_pos + randint(-30, 60), 35), mask=self.animals2imgs[animal_idx])
        self.questions.append("What animal is on top of the roof of the barn?")
        self.answers.append(self.animals[animal_idx])
        
        self.img.paste(self.hay, (self.hay_pos, 125), mask=self.hay)
        animal_idx = randint(len(self.animals))
        self.img.paste(self.animals2imgs[animal_idx], (self.hay_pos + randint(-30, 60), 150), mask=self.animals2imgs[animal_idx])
        self.questions.append("What animal is near the hay?")
        self.answers.append(self.animals[animal_idx])

        self.save_sample()
        
    def show(self):
        self.img.show()

    def save_sample(self):
        self.img = self.img.convert('RGB')
        self.img.save(self.path2imgs + "/sample_{}.jpg".format(self.sample_idx))
        with open(self.path2questions, mode='a') as f:
            for question in self.questions:
                f.write("{},".format(question))
            f.write("\n")
        with open(self.path2answers, mode='a') as f:
            for answer in self.answers:
                f.write("{},".format(answer))
            f.write("\n")
        self.sample_idx += 1
        self.clear_img()



    
def vqa_data(n_samples):
    starttime = time.time()
    animals = ["chicken", "sheep", "cow", "pig"]
    animals2imgs = [Image.open("base_images/{}.png".format(animal)).resize((75, 75)) for animal in animals]
    barn_img = Image.open("base_images/barn.png").resize((150, 150))
    hay_img = Image.open("base_images/hay.png").resize((80, 100))
    img_dim = 256

    # make directories
    for directory in ("data", "data/images"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    gen = VQAGenerator(animals, animals2imgs, barn_img, hay_img, img_dim, n_samples)
    while gen.sample_idx < gen.n_samples:
        gen.make_sample()
    print("VQA Dataset of {} samples generated in {:.1f} seconds.".format(n_samples, time.time() - starttime))

    
