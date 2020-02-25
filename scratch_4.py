import os
import imageio

#jpg_dir = '../saves/png/'
jpg_dir = '/home/omar/PycharmProjects/mlmOK_Pytorch/models/20200213_215704/out/Sample_1_Test_Prediction'
images = []
for file_name in os.listdir(jpg_dir):
    if file_name.endswith('.jpg'):
        file_path = os.path.join(jpg_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(os.path.join(jpg_dir, 'test.gif'), images, fps=5)