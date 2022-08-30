from models import ShuffleNet_v2_30
from models import ResNet18_30
from models import Densenet121_30
from models import Mobilenet_v2_30
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from PIL import ImageFile
from utils.imageNet_datasets import generate_ADV_datasets
import torchattacks as ta
from tqdm import tqdm
from  torchvision import utils as vutils
from config import *

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.enabled = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

def generate_datasets(_method,target_class=None,use_cuda = True):
    # target没有做
    if _method == "PGD":
        atk = ta.PGD(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "FGSM":
        atk = ta.FGSM(target_model, eps=8 / 255)
    elif _method == "CW":
        atk = ta.CW(target_model, c=1e-2, kappa=0, steps=500, lr=0.03)
    elif _method == "BIM":
        atk = ta.BIM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "TIFGSM":
        atk = ta.TIFGSM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "MIFGSM":
        atk = ta.MIFGSM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "DIFGSM":
        atk = ta.DIFGSM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)

    if target_class != None:
        atk.set_target_class(target_class)

    for j,item in enumerate(dataloader):
        x,y = item
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        advs = atk(x,y)
        for i in range(len(y)):
            vutils.save_image(advs[i].cpu(), save_datasets_dir+"/advs/{}_{}_{}.jpg".format(j,i,y[i]), normalize=False) #保存对抗样本
            vutils.save_image(x[i].cpu(), save_datasets_dir + "/ori/{}_{}_{}.jpg".format(j,i,y[i]), normalize=False) #保存干净样本

def get_classifier(model_name,use_cuda=True):

    if model_name == 'ResNet18':
        model = ResNet18_30()
    elif model_name == 'ShuffleNetv2':
        model = ShuffleNet_v2_30()
    elif model_name == 'MobileNetv2':
        model = Mobilenet_v2_30()
    elif model_name == 'DenseNet121':
        model = Densenet121_30()

    model_dir = model_root_dir + '/{}.pt'.format(model_name)

    if model_name:
        print("=====>>>load pretrained model {} from {}".
              format(model_name, model_dir))
        model.load_state_dict(torch.load(
            model_dir,map_location='cuda' if use_cuda else 'cpu'))
        return model
    else:
        return None


def initial_datasets_dir(target_model_name,attack_method):
    os.chdir(train_datasets_dir)
    if not os.path.exists(target_model_name):
        os.mkdir(target_model_name)
    os.chdir(target_model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    if not os.path.exists('ori'):
        os.mkdir('ori')
    if not os.path.exists('advs'):
        os.mkdir('advs')
    path = os.getcwd()
    os.chdir(root_dir)
    return path


if __name__ == "__main__":
    # [PGD,FGSM,CW,TIFGSM,MIFGSM,DIFGSM,BIM]
    #加载ImageNet数据集
    dataloader = DataLoader(
        generate_ADV_datasets(
            benign_dataset,
            transform=train_transform),
        generate_data_batch_size,shuffle=True)

    target_class = None
    for target_model_name in tqdm(["ResNet18","ShuffleNetv2"]):
        target_model = get_classifier(target_model_name).cuda()
        target_model.eval()
        for atk_method in tqdm(["PGD","DIFGSM","FGSM"]):
            save_datasets_dir = initial_datasets_dir(target_model_name,atk_method)
            generate_datasets(target_class=None,_method=atk_method)
