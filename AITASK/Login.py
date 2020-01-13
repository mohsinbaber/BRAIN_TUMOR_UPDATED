from tkinter import *
import pyodbc
from subprocess import call
from tkinter import messagebox
from PIL import Image
import warnings

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')
warnings.filterwarnings("ignore")
root1 = Tk()

img_re2 = Image.open('b20.jpg')
img_re2 = img_re2.resize((700, 400))
img_re2.save("b20.ppm", "ppm")
init_img = PhotoImage(file='b20.ppm')
init_label = Label(image=init_img)
init_label.image = init_img
init_label.place(x=0, y=0)

img_re2 = Image.open('b19.jpg')
img_re2 = img_re2.resize((300, 350))
img_re2.save("b19.ppm", "ppm")
init_img = PhotoImage(file='b19.ppm')
init_label = Label(image=init_img,background="#A0BCE3")
init_label.image = init_img
init_label.place(x=50, y=25)

login = Label(root1, text="System Login", font=(None,17,'underline'))
login.config(width=0, height=0, fg='#FFFFFF',background="#A0BCE3")
login.place(x=130,y=35)

def verify():
    count = -1
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM BrainTumor.dbo.Login')
    result = cursor.fetchall()
    try:
        for row in result:
            if(row[2] == 1): #for admin login
                if(usernameTextbox.get("1.0", "end-1c") == row[0] and passwordTextbox.get() == row[1]):
                    root1.destroy()
                    call(["python","WelcomePage.py"])
                    count = count+1
            elif(row[2] == 2): #for doctor login
                if (usernameTextbox.get("1.0", "end-1c") == row[0] and passwordTextbox.get() == row[1]):
                    root1.destroy()
                    call(["python", "SkullRemoval.py"])
                    count = count + 1
        if (count == -1):
            messagebox.showerror("Login Error", "ID/Username or Password is incorrect!")
    except:
        pass

img_re2 = Image.open('user.png')
img_re2 = img_re2.resize((20, 20))
img_re2.save("userIcon.ppm", "ppm")
init_img = PhotoImage(file='userIcon.ppm')
init_label = Label(image=init_img,background="#A0BCE3")
init_label.image = init_img
init_label.place(x=70, y=103)

username = Label(root1, text="ID/Username:")
username.config(width=0, fg='#FFFFFF', font=44,background="#A0BCE3")
username.place(x=95, y=103)
usernameTextbox = Text(root1, width=16, height=0)
usernameTextbox.place(x=195,y=107)

img_re2 = Image.open('pass.png')
img_re2 = img_re2.resize((20, 20))
img_re2.save("passIcon.ppm", "ppm")
init_img = PhotoImage(file='passIcon.ppm')
init_label = Label(image=init_img,background="#A0BCE3")
init_label.image = init_img
init_label.place(x=70, y=170)


password = Label(root1, text="Password:")
password.config(width=0,  fg='#FFFFFF', font=44,background="#A0BCE3")
password.place(x=95, y=170)
passwordTextbox = Entry(root1, width=22)
passwordTextbox.config(show="*")
passwordTextbox.place(x=195,y=173)

loginBtn = Button(root1, text="Login", height=2,width=12, command=verify)
loginBtn.place(x=110,y=300)

def exit_main():
    root1.destroy()

exitBtn = Button(root1, text="Exit", height=2,width=12, command=exit_main)
exitBtn.place(x=210,y=300)

def reset_pass():
    call(["python", "ResetPassword.py"])

forgotPassLabel = Label(root1, text="Forgot Password?")
forgotPassLabel.config(width=0, fg='#FFFFFF',background="#A0BCE3")
forgotPassLabel.place(x=70,y=240)
forgotPass = Button(root1, text="Click here", height=0,width=0, bd=0,font=(None,9,'underline'))
forgotPass.config(fg='#FFFFFF',background="#A0BCE3", activebackground="#A0BCE3",command=reset_pass)
forgotPass.place(x=167,y=240)


root1.geometry("700x400+350+150")
root1.overrideredirect(True)
root1.configure(background='#008080')
root1.mainloop()