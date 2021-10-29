import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import math
import random
from torchvision import transforms
#################add parameters for IL#############
import timm
import utils
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
#from SwinVideo import SwinTransformer3D
import vision_transformer as vits
#import clip
############################################


class DINONet(nn.Module):
    def __init__(self,arch='vit_small',patch_size=16,n_last_blocks=4,avgpool_patchtokens=False):
        super(DINONet, self).__init__()

        self.n = n_last_blocks
        self.avgpool = avgpool_patchtokens

        self.encoder = vits.__dict__[arch](patch_size, num_classes=0)

        self.classifier_drx = nn.Sequential(LinearClassifier(self.encoder.embed_dim * (n_last_blocks + int(avgpool_patchtokens)), num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dx = nn.Sequential(LinearClassifier(self.encoder.embed_dim * (n_last_blocks + int(avgpool_patchtokens)), num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dy = nn.Sequential(LinearClassifier(self.encoder.embed_dim * (n_last_blocks + int(avgpool_patchtokens)), num_labels=3),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        with torch.no_grad():
            intermediate_output = self.encoder.get_intermediate_layers(x, self.n)
            output = [x[:, 0] for x in intermediate_output]
            if self.avgpool:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1)

        out1 = self.classifier_drx(output)
        out2 = self.classifier_dx(output)
        out3 = self.classifier_dy(output)

        return out1, out2, out3




class DINO_Concat_Net(nn.Module):
    def __init__(self,arch='vit_small',num_frames=8,patch_size=16,n_last_blocks=4,avgpool_patchtokens=False):
        super(DINO_Concat_Net, self).__init__()

        self.n = n_last_blocks
        self.avgpool = avgpool_patchtokens

        self.encoder = vits.__dict__[arch](patch_size, num_classes=0)
        self.num_features = self.encoder.embed_dim * (self.n + int(self.avgpool))*num_frames

        self.classifier_drx = nn.Sequential(LinearClassifier(self.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dx = nn.Sequential(LinearClassifier(self.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dy = nn.Sequential(LinearClassifier(self.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        D = x.shape[2]
        out = []
        for i in range(D):
            intermediate_output = self.encoder.get_intermediate_layers(x[:,:,i,:,:], self.n)
            output = [y[:, 0] for y in intermediate_output]
            if self.avgpool:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1)
            out.append(output)

        assert len(out) == D, "Something is wrong!"
        output = torch.cat(out,dim=1)

        assert self.num_features==output.shape[-1],"Something is wrong!"

        out1 = self.classifier_drx(output)
        out2 = self.classifier_dx(output)
        out3 = self.classifier_dy(output)

        return out1, out2, out3



class TransNet(nn.Module):
    def __init__(self,arch):
        super(TransNet, self).__init__()

        self.encoder = timm.create_model(arch, pretrained=False, num_classes=0)

        self.classifier_drx = nn.Sequential(LinearClassifier(self.encoder.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dx = nn.Sequential(LinearClassifier(self.encoder.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dy = nn.Sequential(LinearClassifier(self.encoder.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.encoder(x)

        out1 = self.classifier_drx(out)
        out2 = self.classifier_dx(out)
        out3 = self.classifier_dy(out)

        return out1, out2, out3

class CLIPmodel(nn.Module):
    def __init__(self,arch):
        super(CLIPmodel, self).__init__()

        self.encoder, _ = clip.load(arch, device="cuda")

        self.classifier_drx = nn.Sequential(LinearClassifier(self.encoder.visual.output_dim, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dx = nn.Sequential(LinearClassifier(self.encoder.visual.output_dim, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dy = nn.Sequential(LinearClassifier(self.encoder.visual.output_dim, num_labels=3),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.encoder.encode_image(x).float()

        out1 = self.classifier_drx(out)
        out2 = self.classifier_dx(out)
        out3 = self.classifier_dy(out)

        return out1, out2, out3

class ConcatTransNet(nn.Module):
    def __init__(self,arch):
        super(ConcatTransNet, self).__init__()

        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        self.encoder = ConcatWrapper(self.backbone, 16)

        self.classifier_drx = nn.Sequential(LinearClassifier(self.encoder.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dx = nn.Sequential(LinearClassifier(self.encoder.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dy = nn.Sequential(LinearClassifier(self.encoder.num_features, num_labels=3),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.encoder(x)

        out1 = self.classifier_drx(out)
        out2 = self.classifier_dx(out)
        out3 = self.classifier_dy(out)

        return out1, out2, out3

class ConcatWrapper(nn.Module):
    #Stacked Frames of Shape (B,C,D,H,W)
    def __init__(self, backbone,num_frames):
        super(ConcatWrapper, self).__init__()
        self.backbone = backbone
        self.num_features = (backbone.embed_dim)*num_frames

    def forward(self,x,detach=False):
        D = x.shape[2]
        out = []
        for i in range(D):
            out.append(self.backbone(x[:,:,i,:,:]))

        assert len(out) == D, "Something is wrong!"

        output = torch.cat(out,dim=1)

        if detach:
            output = output.detach()
        #output shape will be torch.Size([B,D*self.embed_dim])

        assert self.num_features==output.shape[-1],"Something is wrong!"

        return output

class VideoTransNet(nn.Module):
    def __init__(self):
        super(VideoTransNet, self).__init__()

        self.backbone = SwinTransformer3D(
            pretrained2d=False, pretrained=None,
            patch_size=(2,4,4), drop_path_rate=0.1,window_size=(8,7,7)
        )

        self.encoder = TPWrapper(self.backbone)

        self.classifier_drx = nn.Sequential(LinearClassifier(self.encoder.embed_dim, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dx = nn.Sequential(LinearClassifier(self.encoder.embed_dim, num_labels=3),
                                        nn.LogSoftmax(dim=1))
        self.classifier_dy = nn.Sequential(LinearClassifier(self.encoder.embed_dim, num_labels=3),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.encoder(x)

        out1 = self.classifier_drx(out)
        out2 = self.classifier_dx(out)
        out3 = self.classifier_dy(out)

        return out1, out2, out3

class TPWrapper(nn.Module):
    def __init__(self, backbone):
        super(TPWrapper, self).__init__()

        self.backbone = backbone
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.embed_dim = backbone.num_features

    def forward(self,x):
        #shape of x is [B,C,D,H,W]
        out = self.backbone(x)
        out = self.temporal_pool(out)
        out = self.spatial_pool(out)

        out = out.view(out.size(0),-1)

        assert self.embed_dim == out.shape[-1], "something is wrong!"
        #output shape will be torch.Size([B,self.embed_dim])
        return out


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=3):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class TransPolicy():
    def __init__(self,model_type,pretrained_weights,model=None,n_last_blocks=4,avgpool_patchtokens=False,patch_size=16):
        self.img_h = 84
        self.img_w = 84
        self.model_type=model_type
        #### load rgb trained model for IL/BC
        #MODEL_STORE_PATH = '/home/vibhav/projects/robotics/grasp/models/IL/cup_sim/atarinet/train_rand_xyzrcsg/depth/atarinet_depth_28-07-20_4d_rand_xyzrcsg_tr300_e10_cl3_batch32_lr04.ckpt'

        if model_type == 'single':
            self.model = TransNet(model)
        elif model_type == 'concat':
            self.model = ConcatTransNet(model)
        elif model_type == 'swinvideo':
            self.model = VideoTransNet()
        elif model_type == 'dino':
            self.model = DINONet(model,patch_size,n_last_blocks,avgpool_patchtokens)
        elif model_type == 'dino_concat':
            self.model = DINO_Concat_Net(model,16,patch_size,n_last_blocks,avgpool_patchtokens)
        elif model_type == 'clip':
            self.model = CLIPmodel(model)
        else:
            assert False, 'pass a valid model type'

        self.model.cuda()
        utils.restart_from_checkpoint(
        pretrained_weights,
        state_dict=self.model.encoder,
        state_dict_drx=self.model.classifier_drx[0],
        state_dict_dx=self.model.classifier_dx[0],
        state_dict_dy=self.model.classifier_dy[0],
        )

        self.model.eval()

    def rgb2action(self, rgb):
        if self.model_type == 'single' or self.model_type == 'dino' or self.model_type == 'clip':
            # H,W,C -> C,H,W
            rgb = np.transpose(rgb,(2,0,1))
        else:
            # D,H,W,C -> C,D,H,W
            rgb = np.transpose(rgb,(3,0,1,2))

        rgb = rgb.astype(np.float32)/255.0

        img_t = torch.from_numpy(rgb)
        image = torch.unsqueeze(img_t, 0)
        image = transforms.Resize((224,224))(image)
        assert image.shape[-1]==224, "something is wrong"

        #print(image[0, 0, 0])

        if self.model_type == 'single' or self.model_type == 'dino':
            print("came here:")
            image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        #print(image[0, 0, 0])

        image = image.cuda()
        output_drx, output_dx, output_dy = self.model(image)


        _, predicted_drx_gpu = torch.max(output_drx.data, 1)
        predicted_drx_cpu = predicted_drx_gpu.to("cpu")
        predicted_drx = predicted_drx_cpu.numpy()[0]


        _, predicted_dx_gpu = torch.max(output_dx.data, 1)
        predicted_dx_cpu = predicted_dx_gpu.to('cpu')
        predicted_dx = predicted_dx_cpu.numpy()[0]


        _, predicted_dy_gpu = torch.max(output_dy.data, 1)
        predicted_dy_cpu = predicted_dy_gpu.to('cpu')
        predicted_dy = predicted_dy_cpu.numpy()[0]

        new_actions=np.zeros(3, dtype=np.int)

        if (predicted_drx == 0):
            new_actions[0] = 0
        if (predicted_drx == 1):
            new_actions[0] = -1
        if (predicted_drx == 2):
            new_actions[0] = 1


        if (predicted_dx == 0):
            new_actions[1] = 0
        if (predicted_dx == 1):
            new_actions[1] = -1
        if (predicted_dx == 2):
            new_actions[1] = 1


        if (predicted_dy == 0):
            new_actions[2] = 0
        if (predicted_dy == 1):
            new_actions[2] = -1
        if (predicted_dy == 2):
            new_actions[2] = 1

        #drx, dx, dy, dz
        return new_actions

    def depth2action(self, depth):
        depth=depth/100.0
        newdepth=np.zeros([3,84,84])
        newdepth[0,:,:]=depth
        newdepth[1,:,:]=depth
        newdepth[2,:,:]=depth
        newdepth = newdepth.astype(np.float32)

        img_t = torch.from_numpy(newdepth)
        image = torch.unsqueeze(img_t, 0)
        image = transforms.Resize((224,224))(image)
        image = image.to(self.device)
        output_drx, output_dx, output_dy = self.model(image)

        _, predicted_drx_gpu = torch.max(output_drx.data, 1)
        predicted_drx_cpu = predicted_drx_gpu.to("cpu")
        predicted_drx = predicted_drx_cpu.numpy()[0]

        _, predicted_dx_gpu = torch.max(output_dx.data, 1)
        predicted_dx_cpu = predicted_dx_gpu.to('cpu')
        predicted_dx = predicted_dx_cpu.numpy()[0]


        _, predicted_dy_gpu = torch.max(output_dy.data, 1)
        predicted_dy_cpu = predicted_dy_gpu.to('cpu')
        predicted_dy = predicted_dy_cpu.numpy()[0]

        new_actions=np.zeros(3, dtype=np.int)

        if (predicted_drx == 0):
            new_actions[0] = 0
        if (predicted_drx == 1):
            new_actions[0] = -1
        if (predicted_drx == 2):
            new_actions[0] = 1


        if (predicted_dx == 0):
            new_actions[1] = 0
        if (predicted_dx == 1):
            new_actions[1] = -1
        if (predicted_dx == 2):
            new_actions[1] = 1


        if (predicted_dy == 0):
            new_actions[2] = 0
        if (predicted_dy == 1):
            new_actions[2] = -1
        if (predicted_dy == 2):
            new_actions[2] = 1


        return new_actions

    def get_action(self, rgb, depth, rgbdepth):

#        print("came to get action ", rgbdepth)

        if(rgbdepth == 0):  # rgb
            actions = self.rgb2action(rgb)

#        if(rgbdepth == 1): # depth
##        #    print("came here")
#            actions = self.depth2action(depth)

        return actions

