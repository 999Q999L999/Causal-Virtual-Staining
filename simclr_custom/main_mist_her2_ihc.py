import os
import sys
sys.path.append("/data3/lwp/work/pytorch-CycleGAN-and-pix2pix-master/")
import random
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from tqdm import tqdm
from simclr_custom.utils.log import *
from simclr_custom.utils.log import get_logger
from simclr_custom.utils.data import get_cfe_data

from simclr_custom.model.resnet import ResNetSimCLR
from simclr_custom.model.loss_func import sim_loss


def setup_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
def get_scheduler(args, optimizer):
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError("Optimizer not found. Accepted 'Adam' or 'SGD'")
    return scheduler   

def get_optimizer(args, model):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wdecay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
    elif args.optimezer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.wdecay)
    else:
        raise ValueError('Optimizer not found. Accepted "Adam", "SGD"')
    return optimizer
 
def train_one_epoch(model, train_loader, optimizer, loss_func, args):
    model.train()
    device = torch.device("cuda:{}".format(args.device))
    train_loss = []
    logits, labels = [], []
    
    for batch_imgs_0, batch_imgs_1, batch_labels in tqdm(train_loader):
        labels.append(batch_labels)
        batch_imgs_0, batch_imgs_1, batch_labels = batch_imgs_0.to(device), batch_imgs_1.to(device), batch_labels.to(device)
        batch_out_0 = model(batch_imgs_0)
        batch_out_1 = model(batch_imgs_1)
        # batch_logits = torch.cat(batch_logits, dim=0)
        loss, logit = loss_func(batch_out_0, batch_out_1, batch_labels)
        logits.append(logit)
        
        train_loss.append(loss.detach().cpu().item()/len(batch_labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = np.mean(train_loss)
    logits = torch.cat(logits, dim=0).detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    labels = torch.cat(labels, dim=0).numpy()
    
    # train_auc = return_auc_score(train_all_labels, train_all_logits, args.n_classes)
    train_acc = accuracy_score(labels, preds)
    train_f1 = f1_score(labels, preds, average='macro')
    train_auc = roc_auc_score(labels, logits[:,1])
    return avg_loss, train_acc, train_f1, train_auc

def evaluate(model, val_loader, loss_func, args):
    model.eval()
    device = torch.device("cuda:{}".format(args.device))
    logits, labels = [], []
    loss_list = []
    with torch.no_grad():
        for batch_imgs_0, batch_imgs_1, batch_labels in val_loader:
            labels.append(batch_labels)
            batch_imgs_0, batch_imgs_1, batch_labels = batch_imgs_0.to(device), batch_imgs_1.to(device), batch_labels.to(device)
            batch_out_0 = model(batch_imgs_0)
            batch_out_1 = model(batch_imgs_1)
            loss, logit = loss_func(batch_out_0, batch_out_1, batch_labels)
            logits.append(logit)
            loss_list.append(loss.detach().cpu().item())

    avg_loss = np.mean(loss_list)    
    labels = torch.cat(labels, dim=0).numpy()
    logits = torch.cat(logits, dim=0).detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, logits[:,1])
    return avg_loss, acc, f1, auc

def train(args):
    setup_seed(args.seed)
    model_name = "{}_lr_{}_dim_{}_num_sample_{}_seed_{}".format(args.model_type, args.lr, args.out_dim, args.num_train_sample, args.seed)
    log_dir = "{}/logs".format(args.task)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    LOGGER = get_logger("simclr_custom", "{}/logs/{}.txt".format(args.task, model_name))
    tensorboard_dir = "{}/tensorboard/{}".format(args.task, model_name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    
    device=torch.device("cuda:{}".format(args.device))
 
    model = ResNetSimCLR(args.model_type, args.out_dim)
    LOGGER.info("model type: {}, out_dim: {}".format(args.model_type, args.out_dim))
    model = model.to(device)
    
    loss_func = sim_loss
    # loss_func = loss_func.to(device)
    
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    CFE_Data = get_cfe_data(source="mist_her2_ihc")
    train_set = CFE_Data(mode="train", num_samples=args.num_train_sample, pos_ratio=args.pos_ratio)
    val_set = CFE_Data(mode="val", num_samples=args.num_val_sample, pos_ratio=args.pos_ratio)
    LOGGER.info("Train:{}, Test:{}".format(len(train_set),len(val_set)))
    
    train_loader = DataLoader(train_set, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=False)

    best_val_acc = 0.0
    model_save_path = "{}/checkpoints/{}.pth".format(args.task, model_name)
    if not os.path.exists("{}/checkpoints".format(args.task)):
        os.makedirs("{}/checkpoints".format(args.task))
    
    result = {"epoch":[i for i in range(args.max_epoch)],"train_loss":[], "val_loss":[], "train_acc":[], "train_f1":[], "train_auc":[], \
            "val_acc":[], "val_f1":[], "val_auc":[]}

    count = 0
    for epoch in range(args.max_epoch):
        model = model.to(device)
        LOGGER.info("--------------Epoch: {}--------------".format(epoch))
        train_loss, train_acc, train_f1, train_auc = train_one_epoch(model, train_loader, optimizer, loss_func, args)
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["train_f1"].append(train_f1)
        result["train_auc"].append(train_auc)
        LOGGER.info("train loss: {:.8f}, train_acc: {:.4f}, train_f1: {:.4f}, train_auc: {:.4f}".format(train_loss, train_acc, train_f1, train_auc))
        
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, loss_func, args)
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["val_f1"].append(val_f1)
        result["val_auc"].append(val_auc)
        LOGGER.info("val loss: {:.8f}, val_acc: {:.4f}, val_f1: {:.4f}, val_auc: {:.4f}".format(val_loss, val_acc, val_f1, val_auc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            LOGGER.info("save model")
            torch.save(model.to("cpu").state_dict(), model_save_path)
            count = 0
        else:
            count += 1
            if count > args.early_stop:
                break
        
        writer.add_scalars(main_tag="loss", tag_scalar_dict={"train": train_loss, "val": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="acc", tag_scalar_dict={"train": train_acc, "val": val_acc}, global_step=epoch)
        writer.add_scalars(main_tag="f1", tag_scalar_dict={"train": train_f1, "val": val_f1}, global_step=epoch)
        writer.add_scalars(main_tag="auc", tag_scalar_dict={"train": train_auc, "val": val_auc}, global_step=epoch)

        if epoch >= args.schedule_epoch:
            scheduler.step()

    writer.close()
    result = pd.DataFrame(result)
    save_dir = "{}/results".format(args.task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result.to_csv("{}/{}.csv".format(save_dir, model_name)) 
    LOGGER.info("Training finished")
    
def get_params():
    parser = argparse.ArgumentParser(description="All the parameters of this network.")
    parser.add_argument("--task", type=str, default="simclr_mist_her2")
    parser.add_argument("--model_type", type=str, default="resnet18")
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--num_train_sample", type=int, default=5e3)
    parser.add_argument("--num_val_sample", type=int, default=1e3)
    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--max_epoch", type=int, default=100, help="max_epoch")
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.000005, help="lr")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['AdamW', 'Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingLR", choices=[None, 'StepLR', 'CosineAnnealingLR'])
    parser.add_argument('--schedule_epoch', type=int, default=20)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float)
    parser.add_argument('--wdecay', default=1e-3, type=float)
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--num_worker", type=int, default=16, help="num_worker")
    parser.add_argument("--device", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def run():
    args = get_params()
    train(args)
    
if __name__ == "__main__":
    run()