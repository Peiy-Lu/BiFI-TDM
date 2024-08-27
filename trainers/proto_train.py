import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss

def default_train(train_loader,model,optimizer,writer,iter_counter):

    args = model.args
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)
    writer.add_scalar('scale',model.scale.item(),iter_counter)

    avg_class_loss = 0
    avg_anchor_loss = 0
    avg_loss = 0
    avg_acc = 0
    avg_cl_loss = 0

    for i, (inp,_) in enumerate(train_loader):

        iter_counter += 1

        inp = inp.cuda()
        log_prediction, fc1, fc2, M, cl = model(inp)
        anchor_loss1 = F.cross_entropy(fc1[way*model.shots[0]:], target)
        anchor_loss2 = F.cross_entropy(fc2[way*model.shots[0]:], target)
        
        class_loss = criterion(log_prediction,target)

        M = M.sigmoid()
        anchor_loss = M*anchor_loss1 + (1-M)*anchor_loss2
        loss = class_loss + args.gamma_loss*anchor_loss + cl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way

        avg_acc += acc
        avg_loss += loss_value
        avg_class_loss += class_loss.item()
        avg_anchor_loss += anchor_loss.item()
        avg_cl_loss += cl.item()

    avg_acc = avg_acc/(i+1)
    avg_loss = avg_loss/(i+1)
    avg_class_loss = avg_class_loss/(i+1)
    avg_anchor_loss = avg_anchor_loss/(i+1)
    avg_cl_loss = avg_cl_loss/(i+1)

    writer.add_scalar('proto_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)
    writer.add_scalar('class_loss',avg_class_loss,iter_counter)
    writer.add_scalar('anchor_loss',avg_anchor_loss,iter_counter)
    writer.add_scalar('cl_loss',avg_cl_loss,iter_counter)


    return iter_counter,avg_acc

def meta_train(train_loader, model, optimizer, writer, iter_counter):
    model.train()
    args = model.args
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()

    avg_loss = 0
    avg_cls_loss = 0
    avg_aux_loss = 0
    avg_anchor_loss = 0
    avg_acc = 0
    epoch_size = 0
    avg_cl_loss = 0

    for i, (inp, label) in enumerate(train_loader):
        iter_counter += 1
        epoch_size += 1
        inp = inp.cuda()
        label = label.cuda()
        log_prediction, fc1, fc2, M, cl = model(inp=inp)

        anchor_loss1 = F.cross_entropy(fc1[way*model.shots[0]:], target)
        anchor_loss2 = F.cross_entropy(fc2[way*model.shots[0]:], target)
        M = M.sigmoid()
        anchor_loss = M*anchor_loss1 + (1-M)*anchor_loss2

        cls_loss = criterion(log_prediction, target)
       
        loss = cls_loss + args.gamma_loss*anchor_loss +cl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss.item()
        avg_cls_loss += cls_loss.item()
        avg_anchor_loss += anchor_loss.item()


    avg_acc = avg_acc / epoch_size
    avg_cls_loss = avg_cls_loss / epoch_size
    avg_aux_loss = avg_aux_loss / epoch_size
    avg_anchor_loss = avg_anchor_loss / epoch_size
    avg_loss = avg_loss / epoch_size
    avg_cl_loss = avg_cl_loss / epoch_size

    writer.add_scalar('total_loss',avg_loss,iter_counter)
    writer.add_scalar('class_loss',avg_cls_loss,iter_counter)
    writer.add_scalar('anchor_loss',avg_anchor_loss,iter_counter)
    writer.add_scalar('aux_loss',avg_aux_loss,iter_counter)
    writer.add_scalar('cl_loss',avg_cl_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)


    return iter_counter, avg_acc