import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from config import *

# Fichier utiliser pour coder les fonctions "utilitaires", chargement, sauvegarde...


def load_texture(path, size=128): 
    """
    charge une image depuis path, la redimensionne en carré de côté size 
    return tensor (1, 3, H, W) en  [0,1] 
    """
    img = Image.open(path).convert('RGB') 
    img = img.resize((size, size))
    tensor_image = transforms.ToTensor()(img) # la normalisation est déjà faite
    tensor_image = tensor_image.unsqueeze(0) # pour ajouter une dimension batch (1, 3, H, W)
    return tensor_image


def save_image(tensor_image, path):
    """save l'image generee par nca"""
    pil_image = F.to_pil_image(tensor_image.clamp(0, 1)) # clamp par précaution, pour éviter les valeurs négatives ou >1
    pil_image.save(path)


# Pour la partie de fin, si on a envie de tester la reconstruction d'image après l'avoir endommagée. Elle est alors mise dans la boucle d'entrainement, et le modèle doit apprendre à reconstruire l'image endommagée.
def apply_damage(state, rayon):
    """
    efface un cercle au centre de la grille en mettant du bruit a la place
    """
    # TODO
    raise NotImplementedError
