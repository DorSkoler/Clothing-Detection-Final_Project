import os

# Rename all the images in the data folder from 1.jpg to total_amount_of_images.jpg
input_path = '../Data'

index = 1
for f in os.listdir('{}'.format(input_path)):
    os.rename('{}/{}'.format(input_path, f),'{}/{}.jpg'.format(input_path, index))
    index+=1