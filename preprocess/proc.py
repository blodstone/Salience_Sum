import os
import shutil

print('Counting files')
centroid = set([f.split('.')[0] for f in os.listdir('centroid')])
lexpagerank = set([f.split('.')[0] for f in os.listdir('lexpagerank')])
textrank = set([f.split('.')[0] for f in os.listdir('textrank')])
docs = set(os.listdir('docs'))
submodular = set([f.split('.')[0] for f in os.listdir('submodular')])

centroid = docs - centroid
lexpagerank = docs - lexpagerank
textrank = docs - textrank
submodular = docs - submodular

print('Copying files')
all = centroid | lexpagerank | textrank | submodular
for f in all:
    file = f'docs/{f}'
    shutil.copyfile(file, f'missing/{f}')
