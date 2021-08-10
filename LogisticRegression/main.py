import numpy as np
import tkinter as tk
from scipy.special import expit
from PIL import Image, ImageTk

photo = None
photoLabel = None
label = None

def cmd1():
    theta = np.loadtxt("LR-parameters.txt")
    Path = "./Dataset/image/"+ e1.get() 
    global photo,photoLabel,label
    try :
        img = Image.open(Path)
        photo = ImageTk.PhotoImage(img)
        photoLabel = tk.Label(root, image = photo).place(relx=0.07,rely=0.25)
        data = np.array(img.convert('L')).flatten()
        data = np.insert(data, 62500, values=1, axis = 0) / 255.0
        ans = np.around(expit(np.dot(theta.T,data)))
        if ans == 0.0:
            tk.Label(root,text="推测该图片为男性" ).place(relx=0.55, rely=0.5)
        else:
            tk.Label(root,text="推测该图片为女性" ).place(relx=0.55, rely=0.5)
    
    except FileNotFoundError:
        label = tk.Label(root,text="未找到该图片")
        label.place(relx=0.07,rely=0.2)

def cmd2():
    try:
        label.place_forget()

    except:
        pass

if __name__ == '__main__':
    root = tk.Tk()
    root.title("gender_recogniton")
    root.geometry("600x450+450+150")
    
    l1 = tk.Label(root, text="图片的文件名:", font="宋体")
    l1.place(relx=0.03, rely=0.12)

    e1 = tk.Entry()
    e1.place(relx=0.22,rely=0.12)
    b1 = tk.Button(root, text="确定", command=cmd1)
    b2 = tk.Button(root, text="重置", command=cmd2)
    b1.place(relx=0.47, rely=0.11)
    b2.place(relx=0.55, rely=0.11)

    root.mainloop()