import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from config import *

def load_texture(path, size=128):
    """
    charge une image depuis path, la redimensionne en carré de côté size 
    return tensor (1, 3, H, W) en  [0,1] 
    """
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    tensor_image = transforms.ToTensor()(img) # la normalisation est déjà faite
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image


def save_image(tensor_image, path):
    """save l'image generee par nca"""
    pil_image = F.to_pil_image(tensor_image.clamp(0, 1)) #car j'ai du rose chelou des fois
    pil_image.save(path)


#pour la partie de fin tt ca  si on a envie de tester la reconstruction d'image apres damage
def apply_damage(state, rayon):
    """
    efface un cercle au centre de la grille en mettant du bruit a la place
    """
    # TODO
    raise NotImplementedError
