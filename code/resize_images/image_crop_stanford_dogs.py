from PIL import Image
import os.path, sys
import os
import errno

path = "/PATH_TO_DATASET/"
new_path = "/PATH_TO_SAVE_IMAGES/"

dirs = os.listdir(path)

new_width = new_height = 150

for folder in dirs:
    if folder != '.DS_Store':
        ims = os.listdir(os.path.join(path,folder))
        for im in ims:
            current_image = os.path.join(os.path.join(path,folder),str(im))
            #print('Current image: ' + current_image)
            
            im = Image.open(current_image).convert('L')
            
            width, height = im.size
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
            filename_split = len(current_image.split('/'))

            new_filename = '105x105_cropped_' + current_image.split('/')[filename_split - 1]
            #print('current_folder: ' + str(folder))

            if not os.path.exists(os.path.dirname(new_path + str(folder) + '/' + new_filename)):
                try:
                    os.makedirs(os.path.dirname(new_path + str(folder) + '/' + new_filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            #print(new_path + str(folder) + '/' + new_filename)
            im.save(new_path + str(folder) + '/' + new_filename)
            
'''
im = Image.open("test.jpg")

width, height = im.size

new_width = new_height = 150

left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

# Crop the center of the image
im = im.crop((left, top, right, bottom))
im.save('cropped.jpg')
'''
