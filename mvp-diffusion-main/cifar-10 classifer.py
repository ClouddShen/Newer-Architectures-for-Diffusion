from cifar10_models.densenet import densenet121
import numpy as np
import os
import torch
from tqdm import tqdm
from PIL import Image

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2471, 0.2435, 0.2616])
mean = mean[None, :, None, None]
std = std[None, :, None, None]

dir_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
label_dict = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck",
}
for dir in dir_names:
    os.makedirs(os.path.join("class", dir), exist_ok=True)


my_model = densenet121(pretrained=True)
for i in tqdm(range(1000)):
    filename = os.path.join("sampled_images_BigGAN_fixed_299", str(i) + ".jpg")
    with Image.open(filename) as image:
        width, height = image.size
        img = np.array(image)
        img = img[None, :, :, :]
        img = img.transpose(0,3,1,2)
        img = img/255
        img = (img - mean)/std
        img = torch.from_numpy(img).float()

        my_model.eval()
        with torch.no_grad():
            label = label_dict[int(np.argmax(my_model(img)).numpy())]
            image.save(os.path.join("class", label, str(i) + ".jpg"))
