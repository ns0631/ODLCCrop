import cv2, torch, torchvision, os
import numpy as np
from model import UNET

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

print('Loading model...')
model = UNET(in_channels=3, out_channels=1).to('cpu')
last_checkpoint = torch.load('character_backgroundremoval.tar')
model.load_state_dict(last_checkpoint['state_dict'])

while 1:
    filepath = input('Enter image path: ')
    if not os.path.exists(filepath):
        print('Invalid filename.')
    else:
        break

print('Reading/processing image...')
image = cv2.imread(filepath)
x = toTensor(image)

print('Starting inference...')
model.eval()
x = x.to(device='cpu')
with torch.no_grad():
    preds = torch.sigmoid(model(x))
    preds = (preds > 0.5).float()
torchvision.utils.save_image(preds, 'output.png')
print('Prediction written to output.png')
