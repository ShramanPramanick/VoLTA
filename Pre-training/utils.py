from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
import random
from eda_nlp.code.eda import get_only_chars, eda
import warnings
warnings.filterwarnings("ignore")

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class CCImagePairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
             self.transform = transforms.Compose([
                transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

             self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        self.pair_transform = pair_transform

    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform_prime(x)
            return y1, y2
        else:
            return self.transform(x)


class CCTextPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = eda
        else:
            self.transform = get_only_chars

        self.pair_transform = pair_transform

    def __call__(self, x):
        if self.pair_transform is True:
            aug_text = eda(x, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1)
            aug_text_prime = eda(x, alpha_sr=0.1, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.2, num_aug=1)
            y1 = aug_text[0]
            y2 = aug_text_prime[0]
            return y1, y2
        else:
            return self.transform(x)




