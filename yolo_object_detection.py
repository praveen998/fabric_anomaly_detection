from ultralytics import YOLO
import cv2
import datetime
import pandas as pd
import numpy as np
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

#anomaly tracking
unique=[]
count=0


font = cv2.FONT_HERSHEY_SIMPLEX

def printtext(frame,text,org):
    cv2.putText(frame,text,org,font,0.5,(255,0,0),2,cv2.LINE_4)
    return frame
def avg_color(frame,x,y,w,h):
    avg=np.mean(frame[y:y+h,x:x+w])
    return avg.astype(int)
#----------------------------------------------------------------------------
def anomaly_track2(res):
    c_box={}
    for i in range(count,len(res)):
        box=res.boxes[i].xyxy.tolist()[0]
        c_box[i]=box[3]
    print(c_box)
    l=len(c_box)
    while l > 0:
        m=min(c_box.values())
        for key,value in c_box.items():
            if value==m:
                frame=printtext(frame,str(key),(int(box[0]),int(box[1])))


#-----------------------------------------------------------------------------
unique=[]
def anomaly_tracking(box,frame):
    unq=0
    if len(unique)==0:
        unique.append(box)
        print('first box')
        frame=printtext(frame,str(len(unique)),(int(box[0]),int(box[1])))
        return frame
    else:
        xmid=abs(box[2]-box[0])/2
        xmid=int(box[0]+xmid)
        ymid=abs(box[3]-box[1])/2
        ymid=int(box[1]+ymid)
        boxarea=abs(box[2]-box[0])*abs(box[3]-box[1])
        box_color=avg_color(frame,int(box[0]),int(box[1]),int(abs(box[2]-box[0])),int(abs(box[3]-box[1])))
        #print(len(box))
        #print('mid=',mid)
        #print([i for i in unique])
        color={}
        for i in unique:
             ixmid=abs(i[2]-i[0])/2
             ixmid=int(i[0]+ixmid)
             iymid=abs(i[3]-i[1])/2
             iymid=int(i[1]+iymid)
             iarea=abs(i[2]-i[0])*abs(i[3]-i[1])
             i_color=avg_color(frame,int(i[0]),int(i[1]),int(abs(i[2]-i[0])),int(abs(i[3]-i[1])))
             if abs(xmid-ixmid)<100 and abs(ymid-iymid)<200:
                    index=unique.index(i)
                    unique[index]=box
                    if box[1]>400:
                       unique[index]=[999,999,999,999]
                       return frame
                    #color comparison:
                    color[index]=abs(i_color-box_color)
                    continue
                
             else:
                index=unique.index(i)
                if box[1]>400:
                    unique[index]=[999,999,999,999]
                    return frame
                unq+=1
                if unq >=len(unique):
                    if ymid > 200:
                       return frame
                    unique.append(box)
                    frame=printtext(frame,str(len(unique)),(int(box[0]),int(box[1])))
                    return frame
                   # cv2.putText(frame,str(index),(box[0],box[1]),font,0.5,(255,0,0),2,cv2.LINE_4)
                continue
        value= min(color.values())
        for key, val in color.items():
          if val == value:
             print(f'box is similar to {key} the index anomaly')
             frame=printtext(frame,str(key),(int(box[0]),int(box[1])))
             return frame
    return frame
    '''   if unq == len(unique) :
               unique.append(box)
               index=len(unique)
               print('new anomaly added')
        '''
    #cv2.putText(frame, '2', (box[0],box[1]), font, font_size, color, thickness)
    '''unq=0
    if len(unique)==0:
        unique.append(box)
    else:
        mid=abs(box[2]-box[0])//2
        for i in unique:
            if mid < i[2] and mid > i[0] and box[3] > i[3] and box[3] <i[3]+abs(i[3]-i[0]):
                index=unique.index(i)
                unique[index]=box
                #draw index as id of box---------------
                cv2.putText(frame, str(index), (box[0],box[1]), font, font_size, color, thickness)
            else:
                unq+=1
                continue
        if unq == len(unique) :
            unique.append(box)
            index=len(unique)
            #draw index as id of box------------------
            cv2.putText(frame, str(index), (box[0],box[1]), font, font_size, color, thickness)'''
    #return frame


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
    out = cv2.VideoWriter('output_video.mp4',fourcc, 30.0, (None, None)) 
    while cap.isOpened():
        ret,frame=cap.read()
        f_count=f_count+1
        if ret:
          res=detect(frame)
           #DP=result[0]
          # out=video_writer()
          # yield res.plot()
          image=res.plot()
          image=printtext(image,f'count:{len(unique)}',(10,50))
         # anomaly_track2(res)
         
          for i in range(len(res)):
                box=res.boxes[i].xyxy.tolist()[0]
                #----------------------------------------------------------------------------------

                #----------------------------------------------------------------------------------
                image=anomaly_tracking(box,image)
                c=(res.boxes[i].conf).tolist()
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
          cv2.imshow('res',image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
               break
          out.write(res.plot())
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_detetcion('dataset/3.mp4')

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

