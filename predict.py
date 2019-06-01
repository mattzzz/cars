
from fastai.vision import *

defaults.device = torch.device('cpu')

path = '/home/matt/projects/fastai/course-v3/nbs/dl1/data/cars/'
print( path)
filename = 'formula_e/00000071.jpg'
img = open_image(path+filename)

model_path = '.'
learn = load_learner(model_path)

pred_class,pred_idx,outputs = learn.predict(img)
print('{} -> {}'.format(filename, pred_class))



