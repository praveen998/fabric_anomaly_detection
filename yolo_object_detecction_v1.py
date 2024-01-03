from ultralytics import YOLO
import cv2
import datetime
import pandas as pd
#load the pretrained yolo model-------------------
model=YOLO('weight/best.pt')
cls_name=[]
boxs=[]
time=[]
confs=[]
frames=[]

#yolo prediction
def detect(img):
    result=model(img)
    return result[0]


def video_detetcion_live():
    f_count=0
    f=0
    cap=cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))


    while cap.isOpened():
        ret,frame=cap.read()
        f_count=f_count+1
        if ret:
           res=detect(frame)
           #DP=result[0]
          # out=video_writer()
          # yield res.plot()
           cv2.imshow('res',res.plot())
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
           out.write(res.plot())
           for i in range(len(res)):
                box=res.boxes[i].xyxy.tolist()[0]
                area=(box[2]-box[0])*(box[3]-box[1])
                c=(res.boxes[i].conf).tolist()

            # if cal_difference(area,f,c[0]*100,accuracy):
                if int(box[0]/10) != f:
                    boxs.append(box)
                    if res.boxes[i].cls==0:
                        cls_name.append('hole')
                    else:
                        cls_name.append('stain')
                    confs.append(c[0])
                    time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    frames.append(f_count)
                    df = pd.DataFrame({'Time' : time,'frame':frames,'Class_name' : cls_name,'box':boxs,'Conf' : confs}) 
                    df.to_csv('result.csv', index = False)
                    #f = abs(box[2]-box[0])*abs(box[3]-box[1])
                    f = int(box[0]/10)
                else:
                    continue
        else:
            break
               
    #yield res.plot()
    out.release()
    cap.release()
    #out.release()
    cv2.destroyAllWindows()

def video_detetcion(path):
    f_count=0
    f=0
    cap=cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (None, None)) 
    while cap.isOpened():
        ret,frame=cap.read()
        f_count=f_count+1
        if ret:
           res=detect(frame)
           #DP=result[0]
          # out=video_writer()
          # yield res.plot()
           cv2.imshow('res',res.plot())
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
           out.write(res.plot())
           for i in range(len(res)):
                box=res.boxes[i].xyxy.tolist()[0]
                area=(box[2]-box[0])*(box[3]-box[1])
                c=(res.boxes[i].conf).tolist()
            # if cal_difference(area,f,c[0]*100,accuracy):
                if int(box[0]/10) != f:
                    boxs.append(box)
                    if res.boxes[i].cls==0:
                        cls_name.append('hole')
                    else:
                        cls_name.append('stain')
                    confs.append(c[0])
                    time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    frames.append(f_count)
                    df = pd.DataFrame({'Time' : time,'frame':frames,'Class_name' : cls_name,'box':boxs,'Conf' : confs}) 
                    df.to_csv('result.csv', index = False)
                    #f = abs(box[2]-box[0])*abs(box[3]-box[1])
                    f = int(box[0]/10)
                else:
                    continue
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

'''def image_detection(input):
    #img=cv2.imread(input)
    res=detect(input)
    print(len(res))
    for i in range(len(res)):
        box=res.boxes[i].xyxy.tolist()[0]
        print(f'anomaly details:{i}:',box)
        #boxs.append(str(str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])))
        boxs.append(box)
        if res.boxes[i].cls==0:
            print('hole')
            cls_name.append('hole')
        else:
            print('stain')
            cls_name.append('stain')
        print((res.boxes[i].conf).tolist())
        c=(res.boxes[i].conf).tolist()
        confs.append(c[0])
        print('time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df = pd.DataFrame({'Time' : time,'Class_name' : cls_name,'box':boxs,'Conf' : confs}) 
    df.to_csv('result.csv', index = False)
    return df,res.plot()'''

