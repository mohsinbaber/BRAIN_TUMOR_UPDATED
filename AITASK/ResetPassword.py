from tkinter import *
import pyodbc
from tkinter import messagebox

reset = Tk()

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')


resetlab = Label(reset, text="Reset Password", font=(None,20,'underline'))
resetlab.config(width=0, fg='#FFFFFF', background='#008080')
resetlab.place(x=200,y=0)


id = Label(reset,text="ID/Username:", font=(None,15))
id.config(width=0, fg='#FFFFFF', background='#008080')
id.place(x=30,y=80)

idText = Text(reset,width=37,height=0)
idText.place(x=210,y=85)

newpass = Label(reset, text="New Password:",font=(None,15))
newpass.config(width=0, fg='#FFFFFF', background='#008080')
newpass.place(x=30,y=140)

newpassText = Entry(reset, width=49)
newpassText.config(show="*")
newpassText.place(x=210,y=145)

connewpass = Label(reset, text="Confirm Password:",font=(None,15))
connewpass.config(width=0, fg='#FFFFFF', background='#008080')
connewpass.place(x=30,y=200)

connewpassText = Entry(reset, width=49)
connewpassText.config(show="*")
connewpassText.place(x=210,y=205)

def reset_pass():
    if(len(idText.get("1.0","end-1c")) != 0 and len(newpassText.get()) !=0):
                sql = "SELECT * FROM BrainTumor.dbo.Login WHERE Username=?"
                val= (idText.get("1.0", "end-1c"))
                cursor = conn.cursor()
                cursor.execute(sql,val)
                result = cursor.fetchall()
                for row in result:
                    if(row[2] == 1):
                        if(len(newpassText.get()) >= 8):
                            if(newpassText.get() == connewpassText.get()):
                                sql2 = "UPDATE BrainTumor.dbo.Login SET Pass=? WHERE Username=? AND Category=?"
                                val2 = (newpassText.get(),idText.get("1.0", "end-1c"),1)
                                cursor.execute(sql2,val2)
                                conn.commit()
                                messagebox.showinfo("Updating Password","Password updated successfully!")
                            else:
                                messagebox.showerror("Error","Passwords does not match!")
                        else:
                            messagebox.showerror("Error", "Password must be of 8 or more characters!")

                    elif(row[2] == 2):
                        if (len(newpassText.get()) >= 8):
                            if (newpassText.get() == connewpassText.get()):
                                sql2 = "UPDATE BrainTumor.dbo.Login SET Pass=? WHERE Username=? AND Category=?"
                                val2 = (newpassText.get(), idText.get("1.0", "end-1c"), 2)
                                cursor.execute(sql2, val2)
                                conn.commit()

                                sql3 = "UPDATE BrainTumor.dbo.DoctorInfo SET pass=? WHERE ID=?"
                                val3 = (newpassText.get(), idText.get("1.0", "end-1c"))
                                cursor.execute(sql3, val3)
                                conn.commit()
                                messagebox.showinfo("Updating Password", "Password updated successfully!")
                            else:
                                messagebox.showerror("Error", "Passwords does not match!")
                        else:
                            messagebox.showerror("Error", "Password must be of 8 or more characters!")
    else:
        messagebox.showerror("Error","One or more fields are empty!")



resetBtn = Button(reset,text="Reset",height=2,width=20, command=reset_pass)
resetBtn.place(x=230,y=250)

reset.geometry("600x300+400+200")
reset.configure(background='#008080')
reset.resizable(0,0)
reset.mainloop()