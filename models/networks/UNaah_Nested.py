import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from .UNaah import get_backbone, Encoder

def get_skip_in(name, block):
    """ check backbone, defining skip_in for each block """
    if name == 'unet_encoder':
        return [0,0,0,0,0,0,0,0,0,0]
    elif name == 'resnet50':
        return [block*2,block*6,0,block*12,block*2,0,block*24,block*4,block*2,0]

class ConvBlockNested(nn.Module): # Fully connected version

     def __init__(self, ch_in, ch_mid, ch_out=None, use_bn=True, skip_in = 0):
        super(ConvBlockNested, self).__init__()

        ch_out = ch_in/2 if ch_out is None else ch_out
        ch_in = ch_in + skip_in

        # first convolution: either transposed conv, or conv following the skip connection
        
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_mid, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
        self.bn1 = nn.BatchNorm2d(ch_mid) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_mid
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

     def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x
        
        
class Decoder(nn.Module):

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 classes=21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 shortcut_features='default',
                 decoder_use_batchnorm=True,
                 skip_ins = None):
        super(Decoder, self).__init__()
        self.backbone_name = backbone_name
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        
        # build decoder part
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        
        # FCC layers
        print(shortcut_chs)
        print(decoder_filters_in)
        print(decoder_filters)
        

        print('upsample_blocks0_1 in: {}   out: {}'.format(decoder_filters_in[-1]+decoder_filters[-1], decoder_filters[-1]))
        self.conv0_1 = ConvBlockNested(decoder_filters_in[-1]+decoder_filters[-1],decoder_filters[-1],
                                                        decoder_filters[-1],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[0]) # 0_1
                                                       
        print('upsample_blocks1_1 in: {}   out: {}'.format(decoder_filters_in[-2]+decoder_filters[-2],decoder_filters[-2]))                                               
        self.conv1_1 = ConvBlockNested(decoder_filters_in[-2]+decoder_filters[-2],decoder_filters[-2],
                                                        decoder_filters[-2],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[1]) # 1_1
                                                        
        print('upsample_blocks2_1 in: {}   out: {}'.format(decoder_filters_in[-3]+decoder_filters[-3], decoder_filters[-3]))
        self.conv2_1 = ConvBlockNested(decoder_filters_in[-3]+decoder_filters[-3],decoder_filters[-3],
                                                        decoder_filters[-3],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[3]) # 2_1
        print('upsample_blocks3_1 in: {}   out: {}'.format(decoder_filters_in[-4]+decoder_filters[-4], decoder_filters[-4]))
        self.conv3_1 = ConvBlockNested(decoder_filters_in[-4]+decoder_filters[-4],decoder_filters[-4],
                                                        decoder_filters[-4],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[6]) # 3_1                                                
        print('upsample_blocks0_2 in: {}   out: {}'.format(decoder_filters[-1]*2+decoder_filters_in[-1],decoder_filters[-1]))
        self.conv0_2 = ConvBlockNested(decoder_filters[-1]*2+decoder_filters_in[-1],decoder_filters[-1],
                                                        decoder_filters[-1],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[2]) # 0_2
        print('upsample_blocks1_2 in: {}   out: {}'.format(decoder_filters_in[-2]+decoder_filters[-2]*2,decoder_filters[-2]))                                                
        self.conv1_2 = ConvBlockNested(decoder_filters[-2]*2+decoder_filters_in[-2],decoder_filters[-2],
                                                        decoder_filters[-2],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[4]) # 1_2
        print('upsample_blocks2_2 in: {}   out: {}'.format(decoder_filters[-3]*2+decoder_filters_in[-3],decoder_filters[-3]))                                                
        self.conv2_2 = ConvBlockNested(decoder_filters[-3]*2+decoder_filters_in[-3],decoder_filters[-3],
                                                        decoder_filters[-3],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[7]) # 2_2
        print('upsample_blocks0_3 in: {}   out: {}'.format(decoder_filters[-1]*3+decoder_filters_in[-1],decoder_filters[-1]))                                                
        self.conv0_3 = ConvBlockNested(decoder_filters[-1]*3+decoder_filters_in[-1],decoder_filters[-1],
                                                        decoder_filters[-1],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[5]) # 0_3
        print('upsample_blocks1_3 in: {}   out: {}'.format(decoder_filters[-2]*3+decoder_filters_in[-2],decoder_filters[-2]))                                                
        self.conv1_3 = ConvBlockNested(decoder_filters[-2]*3+decoder_filters_in[-2],decoder_filters[-2],
                                                        decoder_filters[-2],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[8]) # 1_3
        print('upsample_blocks0_4 in: {}   out: {}'.format(decoder_filters[-1]*4+decoder_filters[-1],decoder_filters[-1]))
        self.conv0_4 = ConvBlockNested(decoder_filters[-1]*4+decoder_filters_in[-1],decoder_filters[-1],
                                                        decoder_filters[-1],
                                                        use_bn=decoder_use_batchnorm,
                                                        skip_in = skip_ins[9]) # 0_4                                                                                                                                                                                                                                                                                                                                                                                            
        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))
    
    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels   

    def forward(self, x, features):

        x0_0 = features[self.shortcut_features[-4]] # 1st layer
        x1_0 = features[self.shortcut_features[-3]] # 2nd layer
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))
        
        x2_0 = features[self.shortcut_features[-2]] # 3rd layer
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
        
        x3_0 = features[self.shortcut_features[-1]] # 4th layer
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = x # encoder output layer
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        if x0_4.shape[2] == 112:
            x0_4 = self.Up(x0_4)

        x = self.final_conv(x0_4)
        return x

class NestedUNaah(nn.Module):

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 shortcut_features='default',
                 classes=21,
                 decoder_filter_block=64, # has to be 64
                 decoder_use_batchnorm=True):
        super(NestedUNaah,self).__init__()
        skip_ins = get_skip_in(backbone_name, decoder_filter_block)
        decoder_filters = (decoder_filter_block*16,decoder_filter_block*8,decoder_filter_block*4,
        decoder_filter_block*2,decoder_filter_block*1)
        self.encoder = Encoder(backbone_name=backbone_name,
                 pretrained=pretrained,
                 encoder_freeze=encoder_freeze,
                 shortcut_features=shortcut_features)
        self.decoder1 = Decoder(backbone_name=backbone_name,
                 pretrained=pretrained,
                 classes=classes,
                 decoder_filters=decoder_filters,
                 shortcut_features=shortcut_features,
                 decoder_use_batchnorm=decoder_use_batchnorm,
                 skip_ins = skip_ins)
        self.decoder2 = Decoder(backbone_name=backbone_name,
                 pretrained=pretrained,
                 classes=classes,
                 decoder_filters=decoder_filters,
                 shortcut_features=shortcut_features,
                 decoder_use_batchnorm=decoder_use_batchnorm,
                 skip_ins = skip_ins)
    
    def forward(self, x):
        x, features = self.encoder(x)
        out1 = self.decoder1(x, features)
        out2 = self.decoder2(x, features)
        return out1, out2
   
                
if __name__ == "__main__":

    # simple test run
    net = NestedUNaah(backbone_name='resnet50')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(1, 3, 224, 224).normal_()
            targets = torch.empty(1, 21, 224, 224).normal_()
            '''
            out1,out2 = net(batch)
            loss1 = criterion(out1, targets)
            loss1.backward(retain_graph=True)
            loss2 = criterion(out2, targets)
            loss2.backward(retain_graph=True)
            '''
            out = net(batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        #print(out.shape)

    print('fasza.')
