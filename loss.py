import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F
from config import *
import numpy as np

#layers qui nous interessent: indices 1, 6, 11, 18 de vgg.features (cf le papier)
#normalisation ImageNet : mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
#regarder code du challenge ou on faisait du transfer learning pour des inspis c la meme idee

def gram_matrix(f):
    """
    f représente les activations d'une couche du VGG
    f de shape(B, C, H, W) et on return (B, C, C)
    regarder la ref 11 du papier pour la formule (si jai bien compris c'est F*F.T et F est f reshapé en (B, C,HW) pour avoir les memes tailles que le papier, i et j canaux, k pixel dans l'eqt 1)
    
    Inspiré de @leongatys
    """
    N = f.shape[1] # nombre de feature maps
    fm_size = np.array(f.shape[2:]) # shape des feature maps
    F = f.reshape(N,-1) 
    M = np.prod(fm_size)
    G = np.dot(F, F.T) / M
    return G


def get_grams(img):    #sera appele dans main pr calculer les target_grams
    """
    Charge VGG16 pre entraine, le geler, passer img dedans,
    retourne les matrices de Gram aux couches relu1,2,3,4 

    img taille (1, 3, H, W) en [0, 1], on return target_grams (liste de 4 Gram)
    """

    #chargement vgg (modèle pytorch pour commencer?)
    model = tv.vgg16(weights = tv.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
    for p in model.parameters():
        p.requires_grad = False #bloque psk on utilise préentrainé
    target_grams = []
    N = []
    x = img
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in [1, 6, 11, 18]: #relu1,2,3,4
            target_grams.append(gram_matrix(x))
            N.append(x.shape[1])
    return target_grams, N


def texture_loss(y_pred, target_grams, weight=[1., 1., 1., 1.]): #sera appele dans train.py pr calculer la loss
    """
    Calcule la loss entre pred_rgb et target_grams et la retourner

    y_pred: (B, 3, H, W) sortie RGB du NCA en [0,1] (considerer qu'on a deja pris les 3 canaux dans train.py donc qu'on a la bonne entree)
    target_grams: liste de 4 matrices de Gram (precalculees)
    weight: poids pour chaque couche
    appliquer eqt 2 de la ref 11 

    Inspiré de @leongatys
    """
    loss = 0
    pred_grams, N = get_grams(y_pred)
    for l in range(len(target_grams)):
        G = pred_grams[l]
        G_target = target_grams[l]
        loss += float(weight[l])/4 * ((G - G_target)**2).sum() / N[l]**2
    return loss
