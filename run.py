import torch
from torch.autograd import Variable
import sys
from config import *
import os
from loss_func import *
import datetime
from utils.get_trainingloader import get_loader
from utils.get_pretrained_classifier import get_trained_classifier




def mkdir_for_(save_weight,model_name,attack_method):
    os.chdir(save_weight)
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    os.chdir(model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    cwd_path = os.getcwd()
    os.chdir(root_path)
    return cwd_path

def initial_log(log_root_path,model_name,attack_method):
    log_path = mkdir_for_(log_root_path,model_name,attack_method)
    name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H"))
    filename = log_path + '/' + name + '.log'
    return filename

def test(logger):
    generator.eval()
    sum_ori = 0
    sum_advs = 0
    sum_advs_add_mask = 0
    sum_ori_add_mask = 0
    sum_num = 0

    for iter, (oris, advs, labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        advs_add_mask = (torch.tanh(generator(advs)-advs)+1)/2
        ori_add_mask = (torch.tanh(generator(oris)-oris)+1)/2

        logits_ori, _ = discriminator(oris)
        logits_ori_add_mask, _ = discriminator(ori_add_mask)
        logits_advs, _ = discriminator(advs)
        logits_advs_add_mask, _ = discriminator(advs_add_mask)

        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs_add_mask = torch.argmax(logits_advs_add_mask, dim=1)
        pred_ori_add_mask = torch.argmax(logits_ori_add_mask, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)

        temp_ori = torch.sum(pred_ori == labels)
        temp_advs = torch.sum(pred_advs == labels)
        temp_ori_add_mask = torch.sum(pred_ori_add_mask == labels)
        temp_advs_add_mask = torch.sum(pred_advs_add_mask == labels)

        sum_ori += temp_ori
        sum_advs += temp_advs
        sum_ori_add_mask += temp_ori_add_mask
        sum_advs_add_mask += temp_advs_add_mask
        sum_num += len(labels)

    acc1 = sum_ori / sum_num
    acc2 = sum_advs / sum_num
    acc3 = sum_ori_add_mask / sum_num
    acc4 = sum_advs_add_mask / sum_num

    log = "Test ori_acc: {} advs_acc:{} ori_add_mask_acc:{} advs_add_mask_acc:{}".\
        format(acc1, acc2, acc3, acc4)
    logInfo(log,logger)

    return acc3 + acc4

def set_optimizer(name,parameters,training_lr,training_weight_decay):
    if name == "Adam":
        optimizer = torch.optim.Adam(parameters,
                                      lr=training_lr,
                                      weight_decay=training_weight_decay)
    elif name == "SGD":
        optimizer = torch.optim.SGD(parameters,
                                    lr=training_lr,
                                    weight_decay=training_weight_decay)
    else:
        return None

    return optimizer

def train_for_single(logger):

    best_acc = 0

    optimizer = set_optimizer(optim_name,generator.parameters(),
                              training_lr,training_weight_decay)

    loss_CE = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        discriminator.eval()
        generator.train()
        for iter,(oris,advs,labels) in enumerate(train_loader):

            oris = oris.cuda()
            advs = advs.cuda()
            labels = labels.cuda()

            advs_add_mask = (torch.tanh(generator(advs)-advs)+1)/2
            ori_add_mask = (torch.tanh(generator(oris)-oris)+1)/2
            mask_advs = advs - advs_add_mask
            mask_oris = oris - ori_add_mask

            logits_ori, femap_ori = discriminator(oris)
            logits_ori_add_mask, femap_ori_add_mask = discriminator(ori_add_mask)
            logits_advs, _ = discriminator(advs)
            logits_advs_add_mask, femap_advs_add_mask = discriminator(advs_add_mask)

            pred_ori = torch.argmax(logits_ori,dim=1)
            pred_advs_add_mask = torch.argmax(logits_advs_add_mask, dim=1)
            pred_advs = torch.argmax(logits_advs, dim=1)
            pred_ori_add_mask = torch.argmax(logits_ori_add_mask, dim=1)

            sum_ori = torch.sum(pred_ori == labels)
            sum_advs = torch.sum(pred_advs == labels)
            sum_advs_add_mask = torch.sum(pred_advs_add_mask == labels)
            sum_ori_add_mask = torch.sum(pred_ori_add_mask == labels)

            loss1 = (3 * b / 4) * MidLayerVectorLoss(femap_advs_add_mask, femap_ori) + \
                    (b / 4) * MidLayerVectorLoss(femap_ori_add_mask, femap_ori)

            loss2 = (2 * c / 3) * loss_CE(logits_ori_add_mask, labels) + (c / 3) * loss_CE(logits_advs_add_mask, labels)

            loss3 = 0.5 * d * torch.mean(torch.abs(mask_advs)) + 0.5 * d * torch.mean(torch.abs(mask_oris))

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log = "Epoch {} Iter {} loss1-{:0.3e}  loss2-{:0.3e}  " \
                        "loss3-{:0.3e} sum_ori: {} sum_advs: {} sum_ori_add_mask: {} sum_advs_add_mask: {}".\
                format(epoch,iter,loss1,loss2,loss3,sum_ori,
                       sum_advs,sum_ori_add_mask,sum_advs_add_mask)
            logInfo(log,logger)

        if epoch % save_epoch_step == 0:
            torch.save(generator.state_dict(),
                       weight_path + "/{}-{}.pt".format(generator_name, epoch))

            generator.eval()
            acc = validate(epoch,logger)
            if acc > best_acc:
                best_acc = acc
                torch.save(generator.state_dict(),
                           weight_path + "/{}.pt".format(generator_name))
                log = "Save best validation Generator {} at epoch {}".\
                    format(generator_name,epoch)

                logInfo(log,logger)



def validate(epoch,logger):

    sum_ori = 0
    sum_advs = 0
    sum_advs_add_mask = 0
    sum_ori_add_mask = 0
    sum_num = 0

    for iter, (oris, advs, labels) in enumerate(validation_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
        ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2

        logits_ori, _ = discriminator(oris)
        logits_ori_add_mask, _ = discriminator(ori_add_mask)
        logits_advs, _ = discriminator(advs)
        logits_advs_add_mask, _ = discriminator(advs_add_mask)

        pred_ori = torch.argmax(logits_ori, dim=1)
        pred_advs_add_mask = torch.argmax(logits_advs_add_mask, dim=1)
        pred_ori_add_mask = torch.argmax(logits_ori_add_mask, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)

        temp_ori = torch.sum(pred_ori == labels)
        temp_advs = torch.sum(pred_advs== labels)
        temp_ori_add_mask = torch.sum(pred_ori_add_mask == labels)
        temp_advs_add_mask = torch.sum(pred_advs_add_mask == labels)

        sum_ori += temp_ori
        sum_advs += temp_advs
        sum_ori_add_mask += temp_ori_add_mask
        sum_advs_add_mask += temp_advs_add_mask
        sum_num += len(labels)

    acc1 = sum_ori/sum_num
    acc2 = sum_advs / sum_num
    acc3 = sum_ori_add_mask / sum_num
    acc4 = sum_advs_add_mask / sum_num

    log="Validation Epoch {} ori_acc: {} advs_acc:{} ori_add_mask_acc:{} advs_add_mask_acc:{}".\
        format(epoch,acc1,acc2,acc3,acc4)
    logInfo(log,logger)

    return acc3+acc4

def logInfo(log,logger):
    print(log)
    print(log,file=logger,flush=True)

if __name__=="__main__":

    victim_model_list = ["MobileNetv2"]
    source_attack_list = ["FGSM", "DIFGSM", "MIFGSM","PGD"]
    generator_list = ["ResUNet01"]

    generator_name = r'ResUNet01'
    model_name = r'ShuffleNetv2'
    attack_method = r'DIFGSM'  # FGSM PGD CW TIFGSM DIFGSM BIM

    for generator_name in generator_list:
        for model_name in victim_model_list:
            for attack_method in source_attack_list:
                weight_path = mkdir_for_(save_generator_weight,model_name,attack_method)
                filename = initial_log(log_path,model_name,attack_method)
                discriminator = get_trained_classifier(model_name,feature_map=True).cuda()
                generator = get_Generator_Model(generator_name).cuda()
                logger = open(filename,'w')
                log = "generator_name:{} victim_model:{} atk_method:{} optimizer:{} lr:{} weight_decay:{} batch_size:{}".\
                    format(generator_name,model_name,
                           attack_method,
                           optim_name,
                           training_lr,
                           training_weight_decay,
                           batch_size)
                logInfo(log, logger)
                train_loader,validation_loader,test_loader = \
                    get_loader(model_name,attack_method)
                train_for_single(logger)
                test(logger)
