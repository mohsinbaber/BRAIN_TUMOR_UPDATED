from tkinter import *
import tkinter.ttk as new
from tkinter import messagebox
from PIL import Image
from subprocess import call

root = Tk()

img_re2 = Image.open('hos_logo_SkullRemoval.jpg')
img_re2 = img_re2.resize((90, 90))
img_re2.save("hos_logo_SkullRemoval", "ppm")
init_img = PhotoImage(file='hos_logo_SkullRemoval.ppm')
init_label = Label(image=init_img)
init_label.image = init_img
init_label.place(x=100, y=5)

text = Label(root, text="Department of Neurology and Neuroscience", font=(None,20,'underline'))
text.place(x=200,y=0)

text = Label(root, text="Brain Tumor Detection System", font=(None,20,'underline'))
text.place(x=280,y=50)

separator = new.Separator(root,orient="vertical")
separator.place(x=195,y=120,height=520)

separator2 = new.Separator(root,orient="horizontal")
separator2.place(x=5,y=120,width=192)

separator3 = new.Separator(root,orient="vertical")
separator3.place(x=5,y=120,height=520)

separator4 = new.Separator(root,orient="horizontal")
separator4.place(x=5,y=640,width=192)

welcome = Label(root,text="DASHBOARD",font=(None,18))
welcome.place(x=25,y=140)

def on_enter(e):
    add['background'] = '#F0F0F0'

def on_leave(e):
    add['background'] = '#CBCBCB'

def open_add():
    call(["python", "DoctorAddAdmin.py"])

add = Button(root,text="Add Doctor Info",width=25,height=4,background="#CBCBCB",command=open_add)
add.bind("<Enter>",on_enter)
add.bind("<Leave>",on_leave)
add.place(x=9,y=200)

def on_enter(e):
    deld['background'] = '#F0F0F0'

def on_leave(e):
    deld['background'] = '#CBCBCB'

def open_del():
    call(["python", "DoctorDelAdmin.py"])

deld = Button(root,text="Delete Doctor Info",width=25,height=4,background="#CBCBCB",command=open_del)
deld.bind("<Enter>",on_enter)
deld.bind("<Leave>",on_leave)
deld.place(x=9,y=270)

def on_enter(e):
    updd['background'] = '#F0F0F0'

def on_leave(e):
    updd['background'] = '#CBCBCB'

def open_upd():
    call(["python", "DoctorUpdAdmin.py"])

updd = Button(root,text="Update Doctor Info",width=25,height=4,background="#CBCBCB",command=open_upd)
updd.bind("<Enter>",on_enter)
updd.bind("<Leave>",on_leave)
updd.place(x=9,y=340)

def on_enter(e):
    ind['background'] = '#F0F0F0'

def on_leave(e):
    ind['background'] = '#CBCBCB'

def open_info():
    call(["python", "DoctorInformation.py"])

ind = Button(root,text="Doctor Information",width=25,height=4,background="#CBCBCB",command=open_info)
ind.bind("<Enter>",on_enter)
ind.bind("<Leave>",on_leave)
ind.place(x=9,y=410)

def disable_event():
    if(messagebox.askyesno("Logging Out","Do you wish to log out?")):
        root.destroy()
        call(["python", "Login.py"])
    else:
        pass

def on_enter(e):
    abtus['background'] = '#F0F0F0'

def on_leave(e):
    abtus['background'] = '#CBCBCB'

def abt_us():
    messagebox.showinfo("About Us", "This application segments and detect brain tumors in an MRI Images.")

abtus = Button(root,text="About Us",width=25,height=4,background="#CBCBCB",command=abt_us)
abtus.bind("<Enter>",on_enter)
abtus.bind("<Leave>",on_leave)
abtus.place(x=9,y=480)

def on_enter(e):
    lod['background'] = '#F23838'

def on_leave(e):
    lod['background'] = '#CBCBCB'


lod = Button(root,text="Log Out",width=25,height=4,background="#CBCBCB",command=disable_event)
lod.bind("<Enter>",on_enter)
lod.bind("<Leave>",on_leave)
lod.place(x=9,y=550)


separator5 = new.Separator(root,orient="horizontal")
separator5.place(x=195,y=120,width=687)

separator6 = new.Separator(root,orient="vertical")
separator6.place(x=880,y=120,height=520)

separator7 = new.Separator(root,orient="horizontal")
separator7.place(x=195,y=640,width=687)

welcome_msg = Label(root,text="WELCOME, ADMIN!",font=(None,25,'underline'))
welcome_msg.place(x = 380,y=300)

welcome_msg2 = Label(root,text="You can now ADD, DELETE, UPDATE Doctor Information.\nYou can also view Doctors Information from Dashboard.",font=(None,15))
welcome_msg2.place(x=300,y=350)

root.geometry("900x700")
root.protocol("WM_DELETE_WINDOW", disable_event)
root.resizable(0,0)
root.mainloop()