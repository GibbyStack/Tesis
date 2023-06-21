import os
import shutil
import random
import pandas as pd

path = 'Kylberg' # Original path
new_path = 'img_dir' # New path
folders = os.listdir(path) # Folder list

df_train = pd.DataFrame(columns=['images', 'labels']) # Training data frame
df_test = pd.DataFrame(columns=['images', 'labels']) # Testing data frame

target, count_train, count_test = 0, 0, 0
for folder in folders:
    if folder != '.DS_Store':
        print(f' {folder} '.center(50, '='))
        folder_path = os.path.join(path, folder) # Path with the images folder
        images = os.listdir(folder_path) # Images list
        idx_images = list(range(len(images))) # Image index list
        random.shuffle(idx_images) # Shuffle the list of image indexes
        size_train = int(len(images) * 0.8) # Training dataset size
        k = 0
        for idx in idx_images:
            image = images[idx] # Image
            image_path = os.path.join(folder_path, image) # Full image path
            img_path = os.path.join(new_path, image) # New path for image
            shutil.copy(image_path, img_path) # Copy image to new path
            if k <= size_train:
                # Training data
                df_train.loc[count_train]=[image, target]
                count_train += 1
            if k > size_train:
                # Testing data
                df_test.loc[count_test]=[image, target]
                count_test += 1
            k += 1
        target += 1

# Save data frame in .csv
df_train.to_csv('annotations_train.csv', header=False, index=False)
df_test.to_csv('annotations_test.csv', header=False, index=False)