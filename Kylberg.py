import os
import pandas as pd
import shutil

path = './kylberg'
folders = os.listdir(path)

df = pd.DataFrame(columns=['images', 'labels'])

target = 0
count = 0
for folder in folders:
    if folder != '.DS_Store':
        folder_path = os.path.join(path, folder)
        images = os.listdir(folder_path)
        for image in images:
            df.loc[count] = [image, target]
            # image_path = os.path.join(folder_path, image)
            # img_path = os.path.join(path, image)
            # shutil.copy(image_path, img_path)
            count += 1
        target += 1
        # images = os.listdir()
df.to_csv('annotations.csv', header=False, index=False)