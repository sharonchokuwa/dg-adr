import algorithms
import os
from utils.validate import *
from utils.args import *
from utils.misc import *
from utils.results_to_excel import *
from dataset.data_manager import get_dataset
from tqdm import tqdm
from clearml import Task
from datetime import datetime

def remove_current_element(input_list, curr_element):
    return [element for element in input_list if element != curr_element]

def run_algo(target_domain, source_domains, cfg, timestamp, dataset_num, excel_data, args):
    log_path = os.path.join(f'./result/{timestamp}/{cfg.ALGORITHM}/{target_domain[0]}', cfg.OUTPUT_PATH)

    #datasets args
    cfg.DATASET.SOURCE_DOMAINS = source_domains
    cfg.DATASET.TARGET_DOMAINS = target_domain
    cfg.DATASET.NUM_SOURCE_DOMAINS = len(source_domains)
    cfg.TIMESTAMP = timestamp

    #setup clearML
    if not cfg.DEBUG:
        task = Task.init(project_name=f'{cfg.PROJECT_NAME}/{timestamp}', task_name=cfg.DATASET.TARGET_DOMAINS[0])
         
    # init
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log(args, cfg, log_path, len(train_loader), dataset_size, timestamp)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.cuda()

    # train
    if cfg.ALGORITHM == 'GDRNet':
        scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)

    best_performance = 0.0
    iterator = tqdm(range(cfg.EPOCHS))

    for i in iterator:
        epoch = i + 1
        loss_avg = LossCounter()
        loss_alignment_avg = LossCounter()
        loss_focal_avg = LossCounter()

        for data_tuple in train_loader:
            algorithm.train()
            if cfg.ALGORITHM == 'GDRNet':
                image, mask, label, domain, img_index = data_tuple
                minibatch = [image.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
            else: 
                image, label, domain, img_index, domain_name = data_tuple
                minibatch = [image.cuda(), label.cuda().long(), domain.cuda().long(), domain_name]

            loss_dict_iter = algorithm.update(cfg, minibatch)
            loss_avg.update(loss_dict_iter['loss']) 

            if cfg.ALGORITHM == 'DG_ADR':
                loss_focal_avg.update(loss_dict_iter['focal_loss']) 
                loss_alignment_avg.update(loss_dict_iter['align_loss'])
            
        if cfg.ALGORITHM == 'DG_ADR':
            update_writer(writer=writer, epoch=epoch, loss_avg=loss_avg, loss_focal_avg=loss_focal_avg, loss_alignment_avg=loss_alignment_avg)
        else:
            update_writer(writer=writer, epoch=epoch, loss_avg=loss_avg)
                
        alpha = algorithm.update_epoch(epoch)
        algorithm.saving_last(cfg, log_path)
        if cfg.ALGORITHM == 'GDRNet':
            scheduler.step()
    
        # validation
        if epoch % cfg.VAL_EPOCH == 0:
            val_auc, test_auc, _, _ = algorithm.validate(val_loader, test_loader, writer, cfg)
            if val_auc > best_performance:
                best_performance = val_auc
                algorithm.save_model(log_path)
       
    algorithm.renew_model(log_path)
    _, test_auc, test_acc, test_f1 = algorithm.validate(val_loader, test_loader, writer, cfg)

    metrics = {
                'acc': test_acc,
                'auc': test_auc,
                'f1': test_f1,
              }
    model_num = dataset_num + 1
    excel_data = add_data(excel_data, model_num , metrics)
    num_domains = len(cfg.DATASET.SOURCE_DOMAINS) + 1

    if num_domains ==  model_num:
        save_to_excel(log_path, excel_data, cfg)

    os.mknod(os.path.join(log_path, 'done'))
    writer.close()
    if not cfg.DEBUG:
        task.close()

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    domain_names = ['deepdr', 'idrid', 'rldr', 'fgadr', 'aptos', 'messidor_2', 'ddr']

    excel_data = initialize_excel()
    args = get_args()
    cfg = setup_cfg(args)
    
    for idx, domain in enumerate(domain_names):
        target_domain = [domain]
        source_domains = remove_current_element(domain_names, domain)
        run_algo(target_domain=target_domain, source_domains=source_domains, cfg=cfg, \
                 timestamp=timestamp, dataset_num=idx, excel_data=excel_data, args=args)

   
    

    

    
