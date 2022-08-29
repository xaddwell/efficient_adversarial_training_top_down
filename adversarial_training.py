import torch
from torch.autograd import Variable
import sys
from config import *
import os
import datetime
from utils.get_trainingloader import get_loader
from utils.get_pretrained_classifier import get_trained_classifier

# Loss
def MidLayerVectorLoss(femap1,femap2):
    tensor_vector1, tensor_vector2 = getMidLayerVector(femap1, femap2)
    return torch.nn.MSELoss()(tensor_vector1, tensor_vector2)

def getMidLayerVector(femap1,femap2):
    tensor_vector1 = torch.ones((femap1[0].shape[0], 0)).cuda()
    tensor_vector2 = torch.ones((femap2[0].shape[0], 0)).cuda()
    for fe1,fe2 in zip(femap1,femap2):
        tensor_vector1 = torch.cat([tensor_vector1, fe1.cuda()], 1)
        tensor_vector2 = torch.cat([tensor_vector2, fe2.cuda()], 1)

    return tensor_vector1, tensor_vector2

class Lossfunc(torch.nn.Module):
    def __init__(self,alpha1,alpha2):
        super(Lossfunc,self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.loss_mse = torch.nn.MSELoss()

    def forward(self,pred_ori,pred_advs,labels,midlayer_ori,midlayer_advs):
        term11 = self.loss_ce(pred_ori,labels)
        term12 = self.loss_ce(pred_advs,labels)
        term2 = MidLayerVectorLoss(midlayer_ori,midlayer_advs)
        # print(term11,term12,term2)
        return self.alpha1 * (term11+term12),self.alpha2 * term2

class adversarial_trainig():
    def __init__(self):
        self.loss = Lossfunc(alpha1,alpha2)
        self.optimizer = \
            torch.optim.Adam(classifier.parameters(),
                             weight_decay=training_weight_decay,
                             lr=training_lr)

    def run(self,epochs):
        classifier.train()
        best_acc = 0
        for epoch in range(epochs):
            temp_ori, temp_advs, temp_num = 0, 0, 0
            for iter, (oris, advs, labels) in enumerate(train_loader):
                oris = oris.cuda()
                advs = advs.cuda()
                labels = labels.cuda()

                logits_ori, femap_ori = classifier(oris)
                logits_advs, femap_advs = classifier(advs)
                pred_ori = torch.argmax(logits_ori, dim=1)
                pred_advs = torch.argmax(logits_advs, dim=1)
                temp_ori += torch.sum(pred_ori == labels)
                temp_advs += torch.sum(pred_advs == labels)
                temp_num += len(labels)

                loss1,loss2 = self.loss(logits_ori,logits_advs,
                                        labels,femap_ori,femap_advs)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            log = "Epoch-{} ori_acc:{} advs_acc:{} loss1:{} loss2:{}".\
                format(epoch,temp_ori/temp_num,temp_advs/temp_num,loss1,loss2)
            logInfo(log,logger)
            if (epoch+1) % save_epoch_step == 0:
                classifier.eval()
                acc = validate(epoch,logger)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(classifier.state_dict(),
                               weight_path + "/{}.pt".format(model_name))
                    logInfo("save best at epoch {} with acc {}".format(epoch,best_acc),logger)

                torch.save(classifier.state_dict(),
                           weight_path + "/{}-{}.pt".format(model_name, epoch))
                classifier.train()

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

        logits_ori, _ = classifier(oris)
        logits_advs, _ = classifier(advs)
        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)

        sum_ori += torch.sum(pred_ori == labels)
        sum_advs += torch.sum(pred_advs == labels)
        sum_num += len(labels)

    acc1 = sum_ori / sum_num
    acc2 = sum_advs / sum_num
    logInfo("Test ori_acc: {} advs_acc:{}".format(acc1, acc2),logger)

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
    log="Validation Epoch {} ori_acc: {} advs_acc:{} ".format(epoch,acc1,acc2)
    logInfo(log,logger)

    return acc1 + acc2

def logInfo(log,logger):
    print(log)
    print(log,file=logger,flush=True)

if __name__=="__main__":

    victim_model_list = ["ShuffleNetv2","MobileNetv2"]
    source_attack_list = ["FGSM", "DIFGSM", "MIFGSM","PGD"]
    for model_name in victim_model_list:
        for attack_method in source_attack_list:
            weight_path = mkdir_for_(save_weight,model_name,attack_method)
            filename = initial_log(log_path,model_name,attack_method)
            classifier = \
                get_trained_classifier(model_root_dir=model_root_dir,
                                       model_name = model_name+"_with_allfea",
                                       feature_map=True).cuda()
            logger = open(filename,'w')
            log = "victim_model:{} atk_method:{}".format(model_name, attack_method)
            logInfo(log, logger)
            train_loader,validation_loader,test_loader = \
                get_loader(model_name,attack_method)
            trainer = adversarial_trainig()
            trainer.run(epochs=train_epochs)
            test(logger)
