import torch
import os
import modeling.model_manager as models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.utils import save_image
import numpy as np
from torchvision.transforms.functional import to_pil_image

class DG_Model(torch.nn.Module):
    def __init__(self, cfg, log_path):
        super(DG_Model, self).__init__()
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)

        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def forward(self, x):
        features = self.network(x)
        outputs = self.classifier(features)
        return outputs

def preprocess_image_tensor(tensor_image):
    numpy_image = tensor_image.astype(np.float32) / 255
    return numpy_image

def do_gradcam(minibatch, cfg, log_path, idx):  
    image, label, _ = minibatch 

    save_dir = os.path.join('gradcam_heatmaps_output', cfg.DATASET.TARGET_DOMAINS[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    original_img_name = f'original_{idx}.png'

    model =  DG_Model(cfg, log_path)
    model.cuda()
    model.eval()

    output = model(image)
    _, pred = torch.max(output, 1)

    target_layer = [model.network.layer4[-1]]
    print('label = ', label)
    print('pred = ', pred)

    gradcam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
    grayscale_cam = gradcam(input_tensor=image, targets=[ClassifierOutputTarget(label)])

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    input_image_np = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    np_image = input_image_np.astype(np.float32) / 255
    visualization = show_cam_on_image(np_image, grayscale_cam, use_rgb=True)

    im = Image.fromarray(visualization)
    im.save(os.path.join(save_dir, f'heatmap_{idx}.png'))

    txt_save_dir = os.path.join(save_dir, 'output.txt')
    write_to_txt(txt_save_dir, original_img_name, label, pred)

def write_to_txt(txt_save_dir, original_img_name, true_label, predicted_label):
    with open(txt_save_dir, 'a') as file:
        file.write(f"{original_img_name}, True Label: {true_label}, Predicted Label: {predicted_label} \n")



   
        
       



