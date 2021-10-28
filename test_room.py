import torch
from torchvision.transforms.functional import to_tensor,to_pil_image
from PIL import Image
import os,time


# mf='/Users/owl/workspace/BackgroundMattingV2/model.pth'
# model=torch.jit.load(mf).eval()

from model import MattingBase, MattingRefine
mf='/Users/owl/workspace/BackgroundMattingV2/model_files/pytorch_resnet101.pth'
model_type='mattingfine'
if(model_type == 'mattingbase'):
    model = MattingBase('resnet101')
else:
    model = MattingRefine(
        backbone='resnet101',
        backbone_scale=1/4,
        refine_mode='full',
    )
model = model.to('cpu').eval()
model.load_state_dict(torch.load(mf, map_location='cpu'), strict=False)


bgr=Image.open('root_bg.jpeg')
bgr=to_tensor(bgr).unsqueeze(0)

# if(bgr.size(2)<=2048 and bgr.size(3)<=2048):
#     model.backbone_scale=1/4
#     model.refine_sample_pixels = 80_000
# else:
#     model.backbone_scale=1/9
#     model.refine_sample_pixels = 320_000

def remove_bg(src):
    src=to_tensor(src).unsqueeze(0)
    pha,fgr = model(src,bgr)[:2]
    com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cpu').view(1, 3, 1, 1)
    return to_pil_image(com[0].cpu())


def test_img():
    src = Image.open('root_juge.jpeg')
    ss = time.time()
    print('using : {} secs'.format(time.time()-ss))
    img=remove_bg(src)
    img.show('a')


def test_video():
    import cv2,time
    import numpy as np
    def cv2_to_PIL(frame):
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    def PIL_to_cv2(image):
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    cap=cv2.VideoCapture('/Users/owl/workspace/BackgroundMattingV2/romm_juge.mp4')
    i=0
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video_writer=cv2.VideoWriter('/Users/owl/workspace/BackgroundMattingV2/room_juge_out.mp4',fourcc,17,(1280,720))
    ss = time.time()
    while(True):
        ret,frame=cap.read()
        if(not ret):
            break
        img = cv2_to_PIL(frame)
        out=remove_bg(img)
        out=PIL_to_cv2(out)
        video_writer.write(out)
        i+=1
        if(i%10==0):
            print('processed {} at speed {}'.format(i,i/(time.time()-ss)))
    video_writer.release()
    cap.release()

test_video()


