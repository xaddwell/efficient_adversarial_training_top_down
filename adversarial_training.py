


import torch
from torch.autograd import Variable
import sys
from config import *
import os
import datetime
from utils.get_trainingloader import get_loader
from utils.get_pretrained_classifier import get_trained_classifier


class Lossfunc(torch.nn.Module):
    def __init__(self,alpha1,alpha2):
        super(Lossfunc,self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.loss_mse = torch.nn.MSELoss()

    def forward(self,pred_ori,pred_advs,labels,midlayer_ori,midlayer_advs):
        term1 = self.loss_ce(pred_ori,labels) + self.loss_ce(pred_advs,labels)
        term2 = self.loss_mse(midlayer_ori,midlayer_advs)

        return self.alpha1 * term1 + self.alpha2 * term2




class adversarial_trainig():
    def __init__(self,model):

        self.model = model.cuda()
        self.loss = Lossfunc(alpha1,alpha2)
        self.optimizer = \
            torch.optim.Adam(self.model.parameters,
                             weight_decay=training_weight_decay,
                             lr=training_lr)

    def run(self,epochs):
        self.model.train()
        temp_ori, temp_advs = 0, 0
        for epoch in range(epochs):
            for iter, (oris, advs, labels) in enumerate(train_loader):
                oris = oris.cuda()
                advs = advs.cuda()
                labels = labels.cuda()

                logits_ori, femap_ori = self.model(oris)
                logits_advs, femap_advs = self.model(advs)
                pred_ori = torch.argmax(logits_ori, dim=1)
                pred_advs = torch.argmax(logits_advs, dim=1)
                temp_ori += torch.sum(pred_ori == labels)
                temp_advs += torch.sum(pred_advs == labels)

                loss = self.loss(pred_ori,pred_advs,labels,femap_ori,femap_advs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1) % save_epoch_step == 0:
                torch.save(self.model.state_dict(),
                           weight_path + "/{}-{}.pt".format(model_name, epoch))

            logInfo(log, logger)


def mkdir_for_(save_weight,model_name,attack_method):
    os.chdir(save_weight)
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    os.chdir(model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    cwd_path = os.getcwd()
    os.chdir(root_dir)
    return cwd_path

def initial_log(log_root_path,model_name,attack_method):
    log_path = mkdir_for_(log_root_path,model_name,attack_method)
    name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H"))
    filename = log_path + '/' + name + '.log'
    return filename

def test(logger):
    sum_ori = 0
    sum_advs = 0
    sum_num = 0

    for iter, (oris, advs, labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        logits_ori, _ = discriminator(oris)
        logits_advs, _ = discriminator(advs)

        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)

        sum_ori += torch.sum(pred_ori == labels)
        sum_advs += torch.sum(pred_advs == labels)
        sum_num += len(labels)

    acc1 = sum_ori / sum_num
    acc2 = sum_advs / sum_num

    log = "Test ori_acc: {} advs_acc:{}".\
        format(acc1, acc2)
    logInfo(log,logger)

def validate(epoch,logger):

    sum_ori = 0
    sum_advs = 0
    sum_num = 0

    for iter, (oris, advs, labels) in enumerate(validation_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        logits_ori, _ = discriminator(oris)
        logits_advs, _ = discriminator(advs)

        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)

        sum_ori += torch.sum(pred_ori == labels)
        sum_advs += torch.sum(pred_advs== labels)
        sum_num += len(labels)

    acc1 = sum_ori/sum_num
    acc2 = sum_advs / sum_num

    log="Validation Epoch {} ori_acc: {} advs_acc:{} ".\
        format(epoch,acc1,acc2)
    logInfo(log,logger)

def logInfo(log,logger):
    print(log)
    print(log,file=logger,flush=True)

if __name__=="__main__":

    victim_model_list = ["MobileNetv2"]
    source_attack_list = ["FGSM", "DIFGSM", "MIFGSM","PGD"]

    for model_name in victim_model_list:
        for attack_method in source_attack_list:
            weight_path = mkdir_for_(save_weight,model_name,attack_method)
            filename = initial_log(log_path,model_name,attack_method)
            classifier = get_trained_classifier(model_name,feature_map=True).cuda()
            logger = open(filename,'w')
            log = "victim_model:{} atk_method:{}".format(model_name, attack_method)
            logInfo(log, logger)
            train_loader,validation_loader,test_loader = \
                get_loader(model_name,attack_method)
            trainer = adversarial_trainig(model=classifier)
            test(logger)
