"""
This code collected some methods from DomainBed (https://github.com/facebookresearch/DomainBed) and other SOTA methods.
"""
import os
import logging
import collections
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

import utils.misc as misc
import utils.weighting as weighting
from utils.validate import algorithm_validate
import modeling.model_manager as models
from modeling.losses import DahLoss
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug

from vae_dg import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torchvision.models as tv_models
from backpack import backpack, extend
from backpack.extensions import BatchGrad

ALGORITHMS = [
    'ERM',
    'GDRNet',
    'MixupNet',
    'Fishr',
    'DRGen',
    'DG_ADR', # ours
    'VAE_DG',
    'SelfReg',
    'SD',
    ]

def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW, "adagrad": torch.optim.Adagrad,}
    optim_cls = optimizers[name]
    return optim_cls(params, **kwargs)

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

# focal_loss
def softmax_focal_loss(x, target, gamma=2., alpha=0.25):
    n = x.shape[0]
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num =  float(x.shape[1])
    p = torch.softmax(x, dim=1)
    p = p[range_n, target]
    loss = -(1-p)**gamma*alpha*torch.log(p)
    return torch.sum(loss) / pos_num


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - validate()
    - save_model()
    - renew_model()
    - predict()
    """
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches):
        raise NotImplementedError
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch
    
    def validate(self, val_loader, test_loader, writer):
        raise NotImplementedError
    
    def save_model(self, log_path):
        raise NotImplementedError
    
    def renew_model(self, log_path):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)
        
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)
        self.num_classes = num_classes
        model_updates = [{"params": self.network.parameters()},
                         {"params": self.classifier.parameters()},
                        ]

        # model optimizer
        optim_name = cfg.OPTIMIZER.lower()
        if optim_name == 'sgd':
            self.optimizer = get_optimizer(
                            cfg.OPTIMIZER,
                            model_updates,
                            lr = cfg.LEARNING_RATE,
                            momentum = cfg.MOMENTUM,
                            weight_decay = cfg.WEIGHT_DECAY,
                            nesterov=True)
        elif optim_name in ['adam', 'adamw', 'adagrad']:
            self.optimizer = get_optimizer(
                        cfg.OPTIMIZER,
                        model_updates,
                        lr = cfg.LEARNING_RATE,
                        weight_decay = cfg.WEIGHT_DECAY)
        else: 
            raise ValueError('Wrong name of optimizer given')

    def update(self, cfg, minibatch):
        image, label, domain, domain_name = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss}
    
    def validate(self, val_loader, test_loader, writer, cfg):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss, _, _ = algorithm_validate(self, val_loader, writer, self.epoch, 'val', cfg)
            test_auc, test_loss, test_acc, test_f1 = algorithm_validate(self, test_loader, writer, self.epoch, 'test', cfg)
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss, test_acc, test_f1 = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test', cfg)
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc, test_acc, test_f1
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier(self.network(x))

    def saving_last(self, cfg, log_path):
        torch.save({
                    'epoch': self.epoch,
                    'backbone_model_state_dict': self.network.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                }, 
                os.path.join(log_path, 'last_checkpoint.pth'))

#Our method
class DG_ADR(ERM):
    def __init__(self, num_classes, cfg):
        super(DG_ADR, self).__init__(num_classes, cfg)
        
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)
        self.num_classes = num_classes
        self.k = cfg.K
        self.margin = torch.tensor(cfg.MARGIN).cuda()
        model_updates = [{"params": self.network.parameters()},
                         {"params": self.classifier.parameters()},
                        ]

        optim_name = cfg.OPTIMIZER.lower()
        if optim_name == 'sgd':
            self.optimizer = get_optimizer(
                            cfg.OPTIMIZER,
                            model_updates,
                            lr = cfg.LEARNING_RATE,
                            momentum = cfg.MOMENTUM,
                            weight_decay = cfg.WEIGHT_DECAY,
                            nesterov=True)
        elif optim_name in ['adam', 'adamw', 'adagrad']:
            self.optimizer = get_optimizer(
                        cfg.OPTIMIZER,
                        model_updates,
                        lr = cfg.LEARNING_RATE,
                        weight_decay = cfg.WEIGHT_DECAY)
        else: 
            raise ValueError('Wrong name of optimizer given')

    def alignment_loss(self, features, class_labels, domain_labels):
        batch_size, _ = features.size()

        # cosine distance
        similarities = 1 - F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        # overall hard positive and hard negative distances
        hard_positive_distances = []
        hard_negative_distances = []

        for i in range(batch_size):
            # Positive samples: samples with same class label across all domains and avoid same index reference
            positive_mask = (class_labels.unsqueeze(1) == class_labels[i]).clone()
            positive_mask[i] = False
            positive_distances = similarities[i, positive_mask.squeeze(1)]

            # Negative samples: samples with same domain label but different class
            negative_mask = (domain_labels.unsqueeze(1) == domain_labels[i]) & ~(class_labels.unsqueeze(1) == class_labels[i])
            negative_distances = similarities[i, negative_mask.squeeze(1)]

            # Sort distances to get hard positive and hard negative samples
            if self.k <= len(positive_distances):
                sorted_positive_distances, _ = torch.topk(positive_distances, self.k, largest=False)
            elif self.k > len(positive_distances) and len(positive_distances) != 0:
                sorted_positive_distances, _ = torch.topk(positive_distances, len(positive_distances), largest=False)
            elif len(positive_distances) == 0:
                sorted_positive_distances = None

            if self.k <= len(negative_distances):
                sorted_negative_distances, _ = torch.topk(negative_distances, self.k, largest=False)
            elif self.k > len(negative_distances) and len(negative_distances) != 0:
                sorted_negative_distances, _ = torch.topk(negative_distances, len(negative_distances), largest=False)
            elif len(negative_distances) == 0:
                sorted_negative_distances = None

            if sorted_positive_distances is not None:
                if len(sorted_positive_distances) > 1:
                    average_hard_positive = torch.mean(sorted_positive_distances)
                else:
                    # in case there is only one positive sample
                    if len(sorted_positive_distances) == 1:
                        average_hard_positive = sorted_positive_distances
            else:
                # in case there are no positive samples
                average_hard_positive = torch.tensor(0.0) 
            
            if sorted_negative_distances is not None:
                if len(sorted_negative_distances) > 1:
                    average_hard_negative = torch.mean(sorted_negative_distances)
                else:
                    # in case there is only one hard sample
                    if len(sorted_negative_distances) == 1:
                        average_hard_negative = sorted_negative_distances
            else:
                # in case there are no hard samples
                average_hard_negative = torch.tensor(0.0)

            hard_positive_distances.append(average_hard_positive.reshape(1).cuda())
            hard_negative_distances.append(average_hard_negative.reshape(1).cuda())
          
        overall_hard_positive = torch.cat(hard_positive_distances)
        overall_hard_negative = torch.cat(hard_negative_distances)
        loss = torch.max(torch.zeros(1).cuda(), self.margin + overall_hard_positive - overall_hard_negative)
        return loss.mean()

    # weighted cross entropy for DG
    def weighting_loss(self, output, class_labels, domain_names):
        weight = torch.empty(self.num_classes).uniform_(0, 1).cuda()
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
        sample_weight = weighting.get_sample_weights(domain_names, class_labels)
        loss = criterion(output, class_labels)
        loss = loss * sample_weight
        return loss.mean()

    def update(self, cfg, minibatch):
        image, label, domain, domain_name = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)

        if self.epoch + 1 <= cfg.WARM_UP_EPOCHS:
            focal_loss = cfg.WEIGHT_LOSS_ALPHA * softmax_focal_loss(output, label)
            loss = focal_loss
            align_loss = 0
        else:
            align_loss =  cfg.LOSS_ALPHA * self.alignment_loss(features, label, domain)
            focal_loss = cfg.WEIGHT_LOSS_ALPHA * softmax_focal_loss(output, label)
            loss = focal_loss + align_loss
    
        loss.backward()
        self.optimizer.step()
        return {'loss': loss, 'focal_loss': focal_loss, 'align_loss': align_loss}
        
    
    def validate(self, val_loader, test_loader, writer, cfg):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss, _, _ = algorithm_validate(self, val_loader, writer, self.epoch, 'val', cfg)
            test_auc, test_loss, test_acc, test_f1 = algorithm_validate(self, test_loader, writer, self.epoch, 'test', cfg)
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss, test_acc, test_f1 = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test', cfg)
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc, test_acc, test_f1
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier(self.network(x))

    def saving_last(self, cfg, log_path):
        torch.save({
                    'epoch': self.epoch,
                    'backbone_model_state_dict': self.network.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                }, 
                os.path.join(log_path, 'last_checkpoint.pth'))


class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)
                                    
    def img_process(self, img_tensor, mask_tensor, fundusAug):
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new, img_tensor_ori
    
    def update(self, cfg, minibatch):
        
        image, mask, label, domain = minibatch
        
        self.optimizer.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori = self.network(image_ori)
        features_new = self.network(image_new)
        output_new = self.classifier(features_new)

        loss, loss_dict_iter = self.criterion([output_new], [features_ori, features_new], label, domain)
        
        loss.backward()
        self.optimizer.step()

        return loss_dict_iter
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

        
class MixupNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixupNet, self).__init__(num_classes, cfg)
        self.criterion_CE = torch.nn.CrossEntropyLoss()
    
    def update(self, cfg, minibatch, env_feats=None):
        image, label, domain, _ = minibatch
        self.optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = self.mixup_data(image, label)
        outputs = self.predict(inputs)
        loss = self.mixup_criterion(self.criterion_CE, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        self.optimizer.step()

        return {'loss':loss}
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
class Fishr(ERM):
    def __init__(self, num_classes, cfg):
        super(Fishr, self).__init__(num_classes, cfg)
        
        self.num_groups = cfg.FISHR.NUM_GROUPS

        self.network = models.get_net(cfg)
        self.classifier = extend(
            models.get_classifier(self.network._out_features, cfg)
        )
        self.optimizer = None
        
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True)
            for _ in range(self.num_groups)
        ]  
        self._init_optimizer()
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr = self.cfg.LEARNING_RATE,
            momentum = self.cfg.MOMENTUM,
            weight_decay = self.cfg.WEIGHT_DECAY,
            nesterov=True)
        
    def update(self, cfg, minibatch):
        image, label, domain, _ = minibatch
        #self.network.train()

        all_x = image
        all_y = label
        
        len_minibatches = [image.shape[0]]
        
        all_z = self.network(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.cfg.FISHR.PENALTY_ANNEAL_ITERS:
            penalty_weight = self.cfg.FISHR.LAMBDA
            if self.update_count == self.cfg.FISHR.PENALTY_ANNEAL_ITERS != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True)
            #torch.autograd.grad(outputs=loss,inputs=list(self.classifier.parameters()),retain_graph=True, create_graph=True)
            
        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_groups)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()


# DRGen is built based on Fishr method
class DRGen(Algorithm):
    '''
    Refer to the paper 'DRGen: Domain Generalization in Diabetic Retinopathy Classification' 
    https://link.springer.com/chapter/10.1007/978-3-031-16434-7_61
    
    '''
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        self.optimizer = self.algorithm.optimizer
        
        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        #swad_cls = getattr(swad_module, 'LossValley')
        #swad_cls = LossValley()
        self.swad = LossValley(None, cfg.DRGEN.N_CONVERGENCE, cfg.DRGEN.N_TOLERANCE, cfg.DRGEN.TOLERANCE_RATIO)
        
    def update(self, cfg, minibatch):
        loss_dict_iter = self.algorithm.update(minibatch)
        if self.swad:
            self.swad_algorithm.update_parameters(self.algorithm, step = self.epoch)
        return loss_dict_iter
    
    def validate(self, val_loader, test_loader, writer, cfg):
        swad_val_auc = -1
        swad_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self.algorithm, val_loader, writer, self.epoch, 'val(Fishr)', cfg)
            test_auc, test_loss = algorithm_validate(self.algorithm, test_loader, writer, self.epoch, 'test(Fishr)', cfg)

            if self.swad:
                def prt_results_fn(results):
                    print(results)

                self.swad.update_and_evaluate(
                    self.swad_algorithm, val_auc, val_loss, prt_results_fn
                )
                
                if self.epoch != self.cfg.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer, self.epoch, 'val', cfg)
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch, 'test', cfg)
                    
                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")
                        #break
                    
                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset
            
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
                
        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH , 'test', cfg)
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc,swad_loss))
            
        return swad_val_auc, swad_auc    
        
    def save_model(self, log_path):
        self.algorithm.save_model(log_path)
    
    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)
    
    def predict(self, x):
        return self.swad_algorithm.predict(x)

    

class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, num_classes, cfg):
        super(SD, self).__init__(num_classes, cfg)
        self.sd_reg = cfg.SD_PARAM

    def update(self, cfg, minibatch):
        image, label, domain, _ = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        penalty = (output ** 2).mean()
        loss = F.cross_entropy(output, label)
        objective = loss + self.sd_reg * penalty
        
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}


class SelfReg(ERM):
    '''
    SelfReg: Self-supervised Contrastive Regularization for Domain Generalization
    From: https://arxiv.org/abs/2104.09841
    '''

    def __init__(self, num_classes, cfg):
        super(SelfReg, self).__init__(num_classes, cfg)

        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.network.out_features()
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, cfg, minibatch):
        all_x, all_y, domain, _ = minibatch
        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.network(all_x)
        proj = self.cdpl(feat)
        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class VAE_DG(ERM):
    """    
        Variational Autoencoders for Domain Generalization
    """
    def __init__(self, num_classes, cfg):
        super(VAE_DG, self).__init__(num_classes, cfg)
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
        CNN_embed_dim = 256   # latent dim extracted by 2D CNN
        input_shape = 224
        self.vae_network = ResNet_VAE(input_shape, num_classes, cfg, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, CNN_embed_dim=CNN_embed_dim)
        self.optimizer = get_optimizer(
            cfg.OPTIMIZER.lower(),
            self.vae_network.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

    def update(self, cfg, minibatch):
        all_x, all_y, domain, _ = minibatch
        loss, recon_loss, KLD_loss, y_loss = self.vae_network.loss_function(all_x, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(),}

    def predict(self, x):
        return self.vae_network.classifier(x)