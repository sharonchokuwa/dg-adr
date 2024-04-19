import torch.nn as nn
import torchvision.models as models
import torch

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, args):
        super(ResNetSimCLR, self).__init__()
        
        if args.imagenet_pretrained:
            resnet50_model = models.resnet50(pretrained=True)
            in_features_50 = resnet50_model.fc.in_features
            resnet50_model.fc = nn.Linear(in_features_50, out_dim)

            resnet18_model = models.resnet18(pretrained=True)
            in_features_18 = resnet18_model.fc.in_features
            resnet18_model.fc = nn.Linear(in_features_18, out_dim)
            self.resnet_dict = {"resnet18": resnet18_model, 
                                "resnet50": resnet50_model}
        elif args.st10_pretrained:
            resnet50_model = models.resnet50()
            checkpoint_path = r'/pretrained_weights/resnet50_50-epochs_stl10/checkpoint_0040.pth.tar'
            checkpoint = torch.load(checkpoint_path)

            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            log = resnet50_model.load_state_dict(state_dict, strict=False)
            assert log.missing_keys == ['fc.weight', 'fc.bias']

            in_features_50 = resnet50_model.fc.in_features
            resnet50_model.fc = nn.Linear(in_features_50, out_dim)
            print('resnet50_model.fc = ', resnet50_model.fc)
            self.resnet_dict = { "resnet50": resnet50_model}

        else:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim), 
                                "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

            
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
