import cv2
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




# if(bgr.size(2)<=2048 and bgr.size(3)<=2048):
#     model.backbone_scale=1/4
#     model.refine_sample_pixels = 80_000
# else:
#     model.backbone_scale=1/9
#     model.refine_sample_pixels = 320_000

def remove_bg(src,bgr):
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

def capture_bg(cam):
    ret,frame=cam.read()
    begin=False
    ss=time.time()
    while(ret):
        txt='Click C to capture background' if not begin else 'Ready in : {}'.format(5-int(time.time()-ss))
        cv2.putText(frame,txt,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,126,101),3)
        cv2.imshow('a',frame)
        k=cv2.waitKey(1)
        if(k & 0xFF==ord('c')):
            ss=time.time()
            begin=True
        if(begin and time.time()-ss>5):
            ret,frame = cam.read()
            cv2.destroyAllWindows()
            return frame
        ret,frame = cam.read()

def capture_video(camera,video_file,frame_rate=17,dur=10):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video_writer=cv2.VideoWriter(video_file,fourcc,frame_rate,(1280,720))
    ss = time.time()
    while(time.time()-ss<dur):
        ret,frame=camera.read()
        if(not ret):
            break
        cv2.imshow('b',frame)
        cv2.waitKey(1)
        video_writer.write(frame)
    cv2.destroyAllWindows()

def test_video(bgr,video_file,out_put_file,frame_date=17):
    import cv2,time
    import numpy as np
    def cv2_to_PIL(frame):
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    def PIL_to_cv2(image):
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    if(isinstance(bgr,str)):
        bgr=Image.open(bgr)
    else:
        bgr=cv2_to_PIL(bgr)
    bgr=to_tensor(bgr).unsqueeze(0)

    cap=cv2.VideoCapture(video_file)
    i=0
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video_writer=cv2.VideoWriter(out_put_file,fourcc,frame_date,(1280,720))
    ss = time.time()
    while(True):
        ret,frame=cap.read()
        if(not ret):
            break
        img = cv2_to_PIL(frame)
        out=remove_bg(img,bgr)
        out=PIL_to_cv2(out)
        video_writer.write(out)
        i+=1
        if(i%10==0):
            print('processed {} at speed {}'.format(i,i/(time.time()-ss)))
    video_writer.release()
    cap.release()

def pipe_line():
    camera=cv2.VideoCapture(0)
    bgr=capture_bg(camera)
    input('任意键开始录制')
    video_file='test.mp4'
    capture_video(camera,video_file)
    camera.release()
    print('开始去除背景')
    test_video(bgr,video_file,'test_mt.mp4')




pipe_line()


