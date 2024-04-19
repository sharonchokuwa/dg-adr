import os
from utils.args import *
from dataset.data_manager import get_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, sampler, random_split
from gradcam_model import do_gradcam
from torchvision.transforms.functional import to_pil_image


MAIN_PATH = r"/best_model_output_path"
NUM_SAMPLES = 10

def getsampledloader(dataloader):
    total_images = len(dataloader.dataset)
    sampled_dataset, _ = random_split(dataloader.dataset, [NUM_SAMPLES, total_images - NUM_SAMPLES])
    sampled_dataloader = DataLoader(sampled_dataset, batch_size=1, shuffle=False)
    return sampled_dataloader

def remove_current_element(input_list, curr_element):
    return [element for element in input_list if element != curr_element]

def run_algo(target_domain, source_domains, cfg):
    log_path = os.path.join(f'{MAIN_PATH}/{cfg.ALGORITHM}/{target_domain[0]}', cfg.OUTPUT_PATH) # after DG -GDRNet

    cfg.DATASET.NUM_CLASSES = 5
    cfg.DATASET.NUM_SOURCE_DOMAINS = len(source_domains)
    cfg.DATASET.SOURCE_DOMAINS = source_domains
    cfg.DATASET.TARGET_DOMAINS = target_domain

    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    sampled_testloader =  getsampledloader(test_loader)
    
    for idx, data_tuple in enumerate(sampled_testloader):
        if cfg.ALGORITHM == 'GDRNet':
            image, label, domain, img_index = data_tuple
        else:
            image, label, domain, img_index, domain_name = data_tuple
        minibatch = [image.cuda(), label.cuda().long(), domain.cuda().long()]

        save_dir = os.path.join('gradcam_heatmaps_output', cfg.DATASET.TARGET_DOMAINS[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        original_img_name = f'original_{idx}.png'
        img_to_save = to_pil_image(image.detach().cpu().squeeze())
        img_to_save.save(os.path.join(save_dir, original_img_name))

        do_gradcam(minibatch, cfg, log_path, idx)

    
if __name__ == "__main__":
    args = get_args()
    cfg = setup_cfg(args)
    domain_names = ['deepdr', 'rldr', 'fgadr', 'aptos', 'messidor_2', 'ddr']

    for idx, domain in enumerate(domain_names):
        target_domain = [domain]
        source_domains = remove_current_element(domain_names, domain)
        run_algo(target_domain, source_domains, cfg)

#python gradcam_heatmaps.py --algorithm DG_ADR --desc get_heatmaps
