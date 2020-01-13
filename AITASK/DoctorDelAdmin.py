from tkinter import *
import pyodbc
from tkinter import messagebox
from subprocess import call


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')
root = Tk()

menubar = Menu(root)
root.config(menu=menubar)

def clear_entry(event, entry):
    if(search.get() == 'Search by ID'):
        entry.delete(0, END)
    else:
        pass

def search_data():

        cursor = conn.cursor()
        val = search.get()
        sql = "SELECT fname,lname,addresss,contact,designation,ID FROM BrainTumor.dbo.DoctorInfo WHERE ID=?"
        if(search.get().startswith("NH-") and search.get().endswith("-/NL")):
            cursor.execute(sql, val)
            result = cursor.fetchall()
            fnameText.config(state="normal")
            lnameText.config(state="normal")
            addressText.config(state="normal")
            desText.config(state="normal")
            contactText.config(state="normal")
            idText.config(state="normal")

            fnameText.delete("1.0", "end-1c")
            lnameText.delete("1.0", "end-1c")
            addressText.delete("1.0", "end-1c")
            desText.delete("1.0", "end-1c")
            contactText.delete("1.0", "end-1c")
            idText.delete("1.0", "end-1c")
            for row in result:

                fnameText.insert(END,row[0])
                lnameText.insert(END,row[1])
                addressText.insert(END,row[2])
                contactText.insert(END,row[3])
                desText.insert(END,row[4])
                idText.insert(END,row[5])
                fnameText.config(state="disabled")
                lnameText.config(state="disabled")
                addressText.config(state="disabled")
                desText.config(state="disabled")
                contactText.config(state="disabled")
                idText.config(state="disabled")
        else:
            messagebox.showerror("Error","Doctor ID must always starts with initials \"NH-\" and end with \"-/NL\"")



search = Entry(root, width=54, font=('Verdana',12))
search.place(x=52,y=100)
placeholder = 'Search by ID'
search.insert(0,placeholder)
search.bind("<Button-1>", lambda event: clear_entry(event,search))

srchBtn = Button(root, text="Search", height=0,width=8, command=search_data)
srchBtn.place(x=596,y=100)

adddoc = Label(root, text="Delete Doctor Information", font=(None,20,'underline'))
adddoc.config(width=0, fg='#FFFFFF', background='#008080')

fname = Label(root, text="First Name:", font=(None,12))
fname.config(width=0, fg='#FFFFFF', background='#008080')
fnameText = Text(root, width=25, height=0)
fnameText.place(x=140,y=173)

lname = Label(root, text="Last Name:", font=(None,12))
lname.config(width=0, fg='#FFFFFF', background='#008080')
lnameText = Text(root, width=25, height=0)
lnameText.place(x=455,y=173)

address = Label(root, text="Address:", font=(None,12))
address.config(width=0, fg='#FFFFFF', background='#008080')
addressText = Text(root, width=64, height=0)
addressText.place(x=140,y=253)

contact = Label(root, text="Contact Number:", font=(None,12))
contact.config(width=0, fg='#FFFFFF', background='#008080')
contactText = Text(root, width=59, height=0)
contactText.place(x=180,y=323)


des = Label(root, text="Designation:", font=(None,12))
des.config(width=0, fg='#FFFFFF', background='#008080')
desText = Text(root, width=63, height=0)
desText.place(x=150,y=403)

id =  Label(root, text="Doctor ID:", font=(None,12))
id.config(width=0, fg='#FFFFFF', background='#008080')
idText = Text(root, width=63, height=0)
idText.place(x=150,y=483)

id.place(x=50,y=480)
des.place(x=50, y=400)
contact.place(x=50, y=320)
address.place(x=50,y=250)
lname.place(x=370,y=170)
fname.place(x=50, y=170)
adddoc.place(x=210,y=0)


def delete_data():
    ans = messagebox.askyesno("Deleting data","Are you sure you want to delete this record?")
    if(ans == True):
        try:

                cursor = conn.cursor()
                val = search.get()
                sql = "DELETE FROM BrainTumor.dbo.DoctorInfo WHERE ID=?"
                cursor.execute(sql,val)
                conn.commit()
                sql1 = "DELETE FROM BrainTumor.dbo.Login WHERE Username=?"
                cursor.execute(sql1,val)
                conn.commit()
                messagebox.showinfo("Data deleted","Data deleted successfully!")

                fnameText.config(state="normal")
                lnameText.config(state="normal")
                addressText.config(state="normal")
                desText.config(state="normal")
                contactText.config(state="normal")
                idText.config(state="normal")

                search.delete(0, END)
                fnameText.delete("1.0", "end-1c")
                lnameText.delete("1.0", "end-1c")
                addressText.delete("1.0", "end-1c")
                desText.delete("1.0", "end-1c")
                contactText.delete("1.0", "end-1c")
                idText.delete("1.0", "end-1c")
                search.insert(0,placeholder)

                fnameText.config(state="disabled")
                lnameText.config(state="disabled")
                addressText.config(state="disabled")
                desText.config(state="disabled")
                contactText.config(state="disabled")
                idText.config(state="disabled")

        except:
            messagebox.showerror("Error","Record with input ID is not present!")

def logout():
    res = messagebox.askyesno("Logging Out","Are you sure you want to logout?")
    if(res == True):
        root.destroy()
        call(["python", "Login.py"])
    elif(res == False):
        pass


deleteBtn = Button(root, text="Delete", height=2,width=20, font=(None,10), command=delete_data)

logoutBtn = Button(root, text="Logout", height=2,width=20, font=(None,10), command=logout)

logoutBtn.place(x=400,y=580)
deleteBtn.place(x=160,y=580)

def add_data():
    root.destroy()
    call(["python", "DoctorAddAdmin.py"])

def upd_data():
    root.destroy()
    call(["python", "DoctorUpdAdmin.py"])

def del_data():
    pass

def abt_us():
    messagebox.showinfo("About Delete Doctor Page", "*Doctor ID must always starts with initials \"NH-\" and end with \"-/NL\"")

submenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Doctor Information", menu = submenu)
submenu.add_command(label="Add Doctor Info", command=add_data)
submenu.add_command(label="Delete Doctor Info", command=del_data)
submenu.add_command(label="Update Doctor Info", command=upd_data)

submenu1 = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings",menu=submenu1)
submenu1.add_command(label="Logut",command=logout)


submenu2 = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help",menu=submenu2)
submenu2.add_command(label="About Us",command=abt_us)


def disable_event():
    if (messagebox.askyesno("Logging Out", "Do you wish to log out?")):
        root.destroy()
        call(["python", "Login.py"])
    else:
        pass

root.geometry("710x700")
root.title("Delete Doctor Information")
root.config(background='#008080')
#root.protocol("WM_DELETE_WINDOW", disable_event)
root.resizable(0,0)
root.mainloop()