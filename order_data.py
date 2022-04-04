import os

input_path = '../Test'

index = 1
for f in os.listdir('{}'.format(input_path)):
    os.rename('{}/{}'.format(input_path, f),'{}/{}.jpg'.format(input_path, index))
    index+=1