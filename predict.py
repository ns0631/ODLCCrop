print('Importing libraries...')
import cv2, torch, torchvision, os
import numpy as np
from model import UNET

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

def infer(image, model_path='backgroundremoval.tar'):
    model = UNET(in_channels=3, out_channels=1).to('cpu')
    last_checkpoint = torch.load(model_path)
    model.load_state_dict(last_checkpoint['state_dict'])

    x = toTensor(image)

    model.eval()
    x = x.to(device='cpu')
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

        numpy_preds = preds.numpy()
        return numpy_preds

def floodfill(image, s):
    global grid
    row, column = s
    stack = [ s ]
    grid[row][column] = 1
    total_pixels = 1
    visited = [ s ]

    while len(stack) > 0:
        u = stack[-1]
        del stack[-1]
        row, column = u
        neighbors = list()

        if row + 1 < height:
            neighbors.append([row + 1, column])
        if row - 1 > -1:
            neighbors.append([row - 1, column])
        if column + 1 < width:
            neighbors.append([row, column + 1])
        if column - 1 > -1:
            neighbors.append([row, column - 1])

        for neighbor in neighbors:
            row, column = neighbor
            if sum(image[row][column][:]) == 0 or grid[row][column] == 1:
                continue

            total_pixels += 1
            grid[row][column] = 1
            visited.append([row, column])
            stack.append([row, column])

    return total_pixels, visited

def crop(base, inference, area_threshold):
    for row in range(base.shape[0]):
        for column in range(base.shape[1]):
            if inference[row, column] == 0:
                base[row, column, 0] = 0
                base[row, column, 1] = 0
                base[row, column, 2] = 0

    cv2.imwrite('prefloodfill_crop.png')

    height, width = base.shape[:-1]

    #Threshold for shapes is 10%, for images it's 5%
    threshold = area_threshold

    for row in range(height):
        for column in range(width):
            if sum(base[row][column][:]) > 0 and grid[row][column] == 0:
                area, visited = floodfill(base, [row, column])
                if area < int(image_area * threshold):
                    for pixel in visited:
                        r, c = pixel
                        base[r][c][0] = 0
                        base[r][c][1] = 0
                        base[r][c][2] = 0
                else:
                    print(area, threshold)

    return base

if __name__ == '__main__':
    path = input('Enter path: ')

    print('Reading image...')
    image = cv2.imread(path)

    height = image.shape[0]
    width = image.shape[1]

    print('Making prediction...')
    numpy_preds = np.reshape(infer(image), (height, width))

    grid = np.zeros(height * width).reshape(height, width)
    image_area = height * width

    print('Cropping image...')
    cropped_image = crop(image, numpy_preds, 0.1)

    cv2.imwrite('final_crop.png', cropped_image)
