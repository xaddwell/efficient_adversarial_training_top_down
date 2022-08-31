import os
import sys
import torch
import random
import datetime
from config import *
import torchattacks as ta
from torch.autograd import Variable
from utils.get_trainingloader import get_loader
from utils.imageNet_datasets import train_imageNet_datasets
from utils.get_pretrained_classifier import get_trained_classifier

# Loss
def MidLayerVectorLoss(femap1,femap2,delta):
    tensor_vector1, tensor_vector2 = getMidLayerVector(femap1, femap2)
    delta = delta.unsqueeze(1)
    return torch.nn.MSELoss()(tensor_vector1/delta, tensor_vector2/delta)

def getMidLayerVector(femap1,femap2):
    tensor_vector1 = torch.ones((femap1[0].shape[0], 0)).cuda()
    tensor_vector2 = torch.ones((femap2[0].shape[0], 0)).cuda()
    for fe1,fe2 in zip(femap1,femap2):
        tensor_vector1 = torch.cat([tensor_vector1, fe1.cuda()], 1)
        tensor_vector2 = torch.cat([tensor_vector2, fe2.cuda()], 1)
    return tensor_vector1, tensor_vector2

def CWLoss(logits, target, kappa=-5.):
    target = torch.ones(
        logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(
        torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())
    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(other - real, kappa))

class Lossfunc(torch.nn.Module):
    def __init__(self,alpha1,alpha2):
        super(Lossfunc,self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.loss_mse = torch.nn.MSELoss()

    def forward(self,pred_ori,pred_advs,labels,
                midlayer_ori,midlayer_advs,delta):
        term11 = self.loss_ce(pred_ori,labels)
        term12 = self.loss_ce(pred_advs,labels)
        term2 = MidLayerVectorLoss(midlayer_ori[1:-2],midlayer_advs[1:-2],delta)
        return self.alpha1 * (term11+term12),self.alpha2 * term2


class adversarial_trainig():
    def __init__(self,attack_method):
        self.loss = Lossfunc(alpha1,alpha2)
        self.atk_method = attack_method
        self.optimizer = \
            torch.optim.Adam(classifier.parameters(),
                             weight_decay=training_weight_decay,
                             lr=training_lr)

    def get_atk(self,_method,m,steps):
        if _method == "PGD":
            atk = ta.PGD(classifier, eps=m / 255, alpha=4 / 255, steps=steps)
        elif _method == "FGSM":
            atk = ta.FGSM(classifier, eps=m / 255)
        elif _method == "TIFGSM":
            atk = ta.TIFGSM(classifier, eps=m / 255, alpha=4 / 255, steps=steps)
        elif _method == "MIFGSM":
            atk = ta.MIFGSM(classifier, eps=m / 255, alpha=4 / 255, steps=steps)
        elif _method == "DIFGSM":
            atk = ta.DIFGSM(classifier, eps=m / 255, alpha=4 / 255, steps=steps)

        self.atk = atk

    def atk(self,oris,labels,steps = 10,min=2,max=8):
        _method = self.atk_method
        advs = oris.detach()
        for i in range(len(labels)):
            m = random.randint(min,max)
            self.get_atk(_method,m,steps)
            advs[i] = self.atk(oris[i].unsqueeze(0),labels[i].unsqueeze(0))
        return advs

    def run(self,epochs):
        classifier.train()
        best_acc = 0
        for epoch in range(epochs):
            temp_ori, temp_advs, temp_num = 0, 0, 0
            sum_loss1,sum_loss2 = 0,0
            for iter, (oris, advs, labels) in enumerate(train_loader):
                if random_eplison:
                    advs = self.atk(oris,labels).cuda()
                else:
                    advs = advs.cuda()
                oris = oris.cuda()
                delta = torch.abs(advs-oris).mean([1,2,3]).cuda()
                labels = labels.cuda()
                logits_ori, femap_ori = classifier(oris)
                logits_advs, femap_advs = classifier(advs)
                pred_ori = torch.argmax(logits_ori, dim=1)
                pred_advs = torch.argmax(logits_advs, dim=1)
                sum_ori = torch.sum(pred_ori == labels)
                sum_advs = torch.sum(pred_advs == labels)
                sum_num = len(labels)

                temp_ori += sum_ori
                temp_advs += sum_advs
                temp_num += sum_num

                loss1,loss2 = self.loss(logits_ori,logits_advs,
                                        labels,femap_ori,femap_advs,delta)
                loss = loss1 + loss2
                sum_loss1 += loss1
                sum_loss2 += loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = "Epoch-{} iter-{} ori_acc:{:0.3f} advs_acc:{:0.3f} loss1:{:0.3f} loss2:{:0.3f}". \
                    format(epoch,iter,sum_ori / sum_num,sum_advs / sum_num,loss1,loss2)
                logInfo(log, logger)

            log = "Last Epoch-{} ori_acc:{:0.3f} advs_acc:{:0.3f} loss1:{:0.3f} loss2:{:0.3f}".\
                format(epoch,
                       temp_ori/temp_num,
                       temp_advs/temp_num,
                       sum_loss1/temp_num,
                       sum_loss2/temp_num)

            logInfo(log,logger)
            if (epoch+1) % save_epoch_step == 0:
                classifier.eval()
                acc = validate(epoch,logger)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(classifier.state_dict(),
                               weight_path + "/{}.pt".format(model_name))
                    logInfo("save best at epoch {} with acc {:0.3f}".format(epoch,best_acc),logger)

                torch.save(classifier.state_dict(),
                           weight_path + "/{}-{}.pt".format(model_name, epoch))
                classifier.train()

def mkdir_for_(save_weight,model_name,attack_method):
    os.chdir(save_weight)
    _atk_name = attack_method + "_random_eplison" if random_eplison else attack_method
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    os.chdir(model_name)
    if not os.path.exists(_atk_name):
        os.mkdir(_atk_name)
    os.chdir(_atk_name)
    cwd_path = os.getcwd()
    os.chdir(root_dir)
    return cwd_path

def initial_log(log_root_path,model_name,attack_method):
    log_path = mkdir_for_(log_root_path,model_name,attack_method)
    name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H"))
    filename = log_path + '/' + name + '.log'
    return filename

def test(logger):
    sum_ori,sum_advs,sum_num = 0,0,0
    classifier.eval()
    for iter, (oris, advs, labels) in enumerate(test_loader):
        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        logits_ori, _ = classifier(oris)
        logits_advs, _ = classifier(advs)
        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)

        sum_ori += torch.sum(pred_ori == labels)
        sum_advs += torch.sum(pred_advs == labels)
        sum_num += len(labels)

    acc1 = sum_ori / sum_num
    acc2 = sum_advs / sum_num
    logInfo("Testing ori_acc: {:0.3f} advs_acc:{:0.3f}".format(acc1, acc2),logger)

def validate(epoch,logger):
    classifier.eval()
    sum_ori,sum_advs,sum_num = 0,0,0
    for iter, (oris, advs, labels) in enumerate(validation_loader):
        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        logits_ori, _ = classifier(oris)
        logits_advs, _ = classifier(advs)
        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)
        sum_ori += torch.sum(pred_ori == labels)
        sum_advs += torch.sum(pred_advs== labels)
        sum_num += len(labels)

    acc1 = sum_ori/sum_num
    acc2 = sum_advs / sum_num
    log="Validation Epoch-{} ori_acc: {:0.3f} advs_acc:{:0.3f} "\
        .format(epoch,acc1,acc2)
    logInfo(log,logger)

    return acc1 + acc2

def logInfo(log,logger):
    print(log)
    print(log,file=logger,flush=True)

if __name__=="__main__":

    victim_model_list = ["ResNet18","ShuffleNetv2"]
    source_attack_list = ["PGD","FGSM","DIFGSM"]
    for model_name in victim_model_list:
        for attack_method in source_attack_list:

            weight_path = mkdir_for_(save_weight,model_name,attack_method)
            classifier = get_trained_classifier(
                model_root_dir=model_root_dir,
                model_name = model_name+"_with_allfea",
                feature_map=True).cuda()

            log_path = initial_log(log_root_path, model_name, attack_method)
            logger = open(log_path,'w')
            log = "victim_model:{} atk_method:{} random_eplison:{}".\
                format(model_name, attack_method,random_eplison)
            logInfo(log, logger)

            data_dir = train_datasets_dir + "/" + model_name + "/" + attack_method
            datasets = train_imageNet_datasets(data_dir)
            train_loader,validation_loader,test_loader = \
                get_loader(datasets)

            trainer = adversarial_trainig(attack_method)
            trainer.run(epochs=train_epochs)
            test(logger)
