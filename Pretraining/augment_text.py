import os
import glob
from eda_nlp.code.eda import eda

def augment_text(root_dir):
    for filename in glob.iglob(root_dir + '**/*.txt', recursive=True):
        with open(filename, "r") as f:
            text_data = f.read()
            aug_text_data = eda(text_data, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=100)
        
        aug_filename = filename.split(".")[0] + '_aug_0.1.txt'
        with open(aug_filename, 'w') as filehandle:
            for listitem in aug_text_data:
                filehandle.write('%s\n' % listitem)

if __name__ == '__main__':
    augment_text('/export/io76a/data/shraman/multimodal_self_supervision/datasets/mscoco/mscoco/')



