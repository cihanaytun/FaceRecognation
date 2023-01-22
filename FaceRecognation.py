import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image



window=tk.Tk()
window.iconbitmap("Image\icon.ico")
window.title('YUZ TANIMA')
window.configure(background='silver')
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f"600x500-{(screen_width//2)-280}-{(screen_height//2)-270}")
window.resizable(width=False,height=False)

lblD=tk.Label(text = "YUZ TANIMA",
width = 50, height = 15, fg ="black",  
bg = "silver", font = ('times', 13, 'bold') )  
lblD.place(x = 60, y = -70)

lblMy=tk.Label(text = "CIHAN AYTUN",
width = 30, height = 1, fg ="black",  
bg = "silver", font = ('times', 10, 'underline') )  
lblMy.place(x = -30, y = 470)

lbl=tk.Label(text = "Lutfen ID Girin :",
width = 12, height = 1, fg ="black",  
bg = "silver", font = ('times', 15, ' bold ') )  
lbl.place(x = 95, y = 250)

lblInput=tk.Label(text = "[Bilgi] Lutfen ID numarasını 1 den başlayarak sıra ile giriniz",
width = 55, height = 1, fg ="black",  
bg = "silver", font = ('times', 7, ' bold ') )  
lblInput.place(x = 230, y = 280)

lblName=tk.Label(text = "Lutfen Isim Girin :",  
width = 14, height = 1, fg ="black",  
bg = "silver", font = ('times', 15, ' bold ') )  
lblName.place(x = 77, y = 150)

txt = tk.Entry( 
width = 20, bg ="white",  
fg ="black", font = ('times', 15, ' bold ')) 
txt.place(x = 250, y = 250)

txtName = tk.Entry(  
width = 20, bg ="white",  
fg ="black", font = ('times', 15, ' bold ')) 
txtName.place(x = 250, y = 150)



def faceDetect():
    window.iconbitmap("Image\icon.ico")
    window.wm_attributes("-alpha",0.7)  
    camera = cv2.VideoCapture(0)
    camera.set(3, 440) 
    camera.set(4, 380) 
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_ID=(txt.get())
    if face_ID=="" :
         tk.messagebox.showwarning(" DIKKAT ","Lutfen bir ID numarasi girin")
         window.wm_attributes("-alpha",1)
         tk.destroy(window)        
    if face_ID == '0':
         tk.messagebox.showwarning(" DIKKAT ","Lutfen  ID numarasini 1 ve uzerinde bir rakam girin")
         window.wm_attributes("-alpha",1)
         tk.destroy(window)  
    
    txtPerson=(txtName.get())
    if txtPerson=="":
        tk.messagebox.showwarning(" DIKKAT ","Lutfen isim girin")
        window.wm_attributes("-alpha",1)
        tk.destroy(window)        

    wtiteToTxtFile=open('persons.txt',"a")
    
    wtiteToTxtFile.write(txtPerson+"\n")
    wtiteToTxtFile.close()    
    imageQuantity = 0
    while(True):
   
         save, PersonImage = camera.read()
         gray = cv2.cvtColor(PersonImage, cv2.COLOR_BGR2GRAY)
         faces = face_detect.detectMultiScale(gray, 1.3, 5)
         for (x,y,w,h) in faces:
             cv2.rectangle(PersonImage, (x,y), (x+w,y+h), (255,0,0), 3)
             imageQuantity += 1
      
             cv2.imwrite("Datas/" + str(face_ID) + '.' + str(imageQuantity) + ".jpg", gray[y:y+h,x:x+w]) 
             
             cv2.imshow('Kayit Sayfasi', PersonImage)
         quit = cv2.waitKey(100) & 0xff
        
         if quit == 27: # Exit by pressing Esc key
            break

         elif imageQuantity >= 200:
            break
  
    tk.messagebox.showinfo("MESAJ !!!","\n KAYIT TAMAMLANDI...")
    txt.delete(0,'end')
    txtName.delete(0,'end')
    camera.release()
    cv2.destroyAllWindows()
    window.wm_attributes("-alpha",1)
 
    
 
 
def trainData():      
    dataPath = 'Datas'
    faceRecognize = cv2.face.LBPHFaceRecognizer_create() 
    face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
    def FaceAndTiteToTxtFile(dataPath):
        registredFaces = [os.path.join(dataPath,i) for i in os.listdir(dataPath)]
        faceExamples=[]
        person = []
        
        for facePath in registredFaces:
        
          PIL_image = Image.open(facePath).convert('L')   
          image_np = np.array(PIL_image,'uint8')   
          ID = int(os.path.split(facePath)[-1].split(".")[0])
    
          faces = face_detect.detectMultiScale(image_np) 
        
          for (x,y,w,h) in faces:
              faceExamples.append(image_np[y:y+h,x:x+w])
              person.append(ID)
        return faceExamples,person

    faces,T_person = FaceAndTiteToTxtFile(dataPath)
    faceRecognize.train(faces, np.array(T_person))
    faceRecognize.write('E_Data/E_Datas.yml') 

    lblImageSavedShow=tk.Label(text=""f"\n {len(np.unique(T_person))} Adet yüz egitildi." ,
    width = 14, height = 2, fg ="black",  
    bg = "silver", font = ('times', 13, ' bold ') )  
    lblImageSavedShow.place(x = 215, y = 420)
    lblImageSavedShow.after(4000, lblImageSavedShow.destroy)
   
    
    
    
    
def knowTheFaces():
    window.iconbitmap("Image\icon.ico")
    window.wm_attributes("-alpha",0.7)
    recognizePerson = cv2.face.LBPHFaceRecognizer_create()
    recognizePerson.read('E_Data/E_Datas.yml')
    face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    personNameText = cv2.FONT_ITALIC
    ID=0              
    writeToTxtFile=open('persons.txt','r')
    
    readData=writeToTxtFile.readlines()
    person=['',]
    print(person)
    for i in readData:
      person.append(i)
    writeToTxtFile.close()     
    
    camera = cv2.VideoCapture(0)
    camera.set(3, 440) 
    camera.set(4, 380)  

    while True:
          save, PersonImage = camera.read()
          gri = cv2.cvtColor(PersonImage, cv2.COLOR_BGR2GRAY)
 
          faces = face_detect.detectMultiScale(gri)
          for (x, y, w, h) in faces:
               ID, recognize = recognizePerson.predict(gri[y:y + h, x:x + w]) 
               if (recognize < 100):
                   ID = person[ID]
                   cv2.rectangle(PersonImage, (x, y), (x + w, y + h), (0, 255, 0), 3)       
               else:
                   ID = "TANINMIYOR"
                   cv2.rectangle(PersonImage, (x, y), (x + w, y + h), (0, 0, 255), 3)

               cv2.putText(PersonImage, str(ID), (x + 5, y - 5), personNameText, 1, (255, 255, 255), 2)
               
          cv2.imshow("TANIMA SAYFASI", PersonImage)
              
          quit = cv2.waitKey(100) & 0xff 
          if quit == 27 : # Exit by pressing Esc key
             break

    camera.release()
    cv2.destroyAllWindows() 
    window.wm_attributes("-alpha",1)
    


def Close():
    window.destroy()
 


   
faceSave=tk.Button(text='YUZ  KAYDET',command=faceDetect,fg ="white", bg ="dimgray",  
width = 15, height = 3,font = ('times', 10, ' bold '), activebackground = "cadetblue")
faceSave.place(x=50,y=350)

faceEducate=tk.Button(text='VERILERI  EGIT',command=trainData,fg ="white", bg ="dimgray",  
width = 15, height = 3,font = ('times', 10, ' bold '), activebackground = "cadetblue")
faceEducate.place(x=173,y=350)

faceRecognize=tk.Button(text='YUZ  TANI',command=knowTheFaces,fg ="white", bg ="dimgray",  
width = 15, height = 3,font = ('times', 10, ' bold '), activebackground = "cadetblue")
faceRecognize.place(x=295,y=350)

close=tk.Button(text='ÇIKIŞ',command=Close,fg ="white", bg ="dimgray",  
width = 15, height = 3, font = ('times', 10, ' bold '),activebackground = "cadetblue")
close.place(x=420,y=350)


window.mainloop()
