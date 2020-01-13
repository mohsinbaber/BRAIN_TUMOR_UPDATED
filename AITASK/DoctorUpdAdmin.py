from tkinter import *
import pyodbc
from tkinter import messagebox
from subprocess import call
from tkcalendar import Calendar,DateEntry
from tkinter import ttk
import datetime

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
    sql = "SELECT ID,fname,lname,addresss,contact,gender,designation,dob,doj FROM BrainTumor.dbo.DoctorInfo WHERE ID=?"
    if (search.get().startswith("NH-") and search.get().endswith("-/NL")):
        cursor.execute(sql, val)
        result = cursor.fetchall()
        fnameText.config(state="normal")
        lnameText.config(state="normal")
        addressText.config(state="normal")
        desText.config(state="normal")
        contactText.config(state="normal")
        passwordText.config(state="normal")
        dropdown.config(state="normal")
        conpasswordText.config(state="normal")

        fnameText.delete("1.0", "end-1c")
        lnameText.delete("1.0", "end-1c")
        addressText.delete("1.0", "end-1c")
        desText.delete("1.0", "end-1c")
        contactText.delete("1.0", "end-1c")
        passwordText.delete(0,END)
        tkvar.set('Choose Gender')
        conpasswordText.delete(0, END)
        dayVar.set('Choose Day')
        monVar.set('Choose- Month')
        yearVar.set('Choose Year')
        dojCal.delete(0,END)

        for row in result:
            fnameText.insert(END,row[1])
            lnameText.insert(END,row[2])
            addressText.insert(END,row[3])
            contactText.insert(END,row[4])
            tkvar.set(row[5])
            desText.insert(END,row[6])
            day,month,year = row[7].split("-")
            dayVar.set(day)
            monVar.set(month)
            yearVar.set(year)
            dojCal.insert(END,row[8])
    else:
        messagebox.showerror("Error", "Doctor ID must always starts with initials \"NH-\" and end with \"-/NL\"")


search = Entry(root, width=57, font=('Verdana',12))
search.place(x=52,y=70)
placeholder = 'Search by ID'
search.insert(0,placeholder)
search.bind("<Button-1>", lambda event: clear_entry(event,search))

srchBtn = Button(root, text="Search", height=0,width=8, command=search_data)
srchBtn.place(x=575,y=69)

adddoc = Label(root, text="Update Doctor Information", font=(None,20,'underline'))
adddoc.config(width=0, fg='#FFFFFF', background='#008080')

fname = Label(root, text="First Name:", font=(None,12))
fname.config(width=0, fg='#FFFFFF', background='#008080')
fnameText = Text(root, width=25, height=0)
fnameText.place(x=140,y=123)

lname = Label(root, text="Last Name:", font=(None,12))
lname.config(width=0, fg='#FFFFFF', background='#008080')
lnameText = Text(root, width=25, height=0)
lnameText.place(x=435,y=123)

address = Label(root, text="Address:", font=(None,12))
address.config(width=0, fg='#FFFFFF', background='#008080')
addressText = Text(root, width=62, height=0)
addressText.place(x=140,y=183)

contact = Label(root, text="Contact Number:", font=(None,12))
contact.config(width=0, fg='#FFFFFF', background='#008080')
contactText = Text(root, width=57, height=0)
contactText.place(x=180,y=253)

gender = Label(root, text="Gender:", font=(None,12))
gender.config(width=0, fg='#FFFFFF', background='#008080')

choices = {'Male', 'Female','Others'}
tkvar = StringVar(root)
tkvar.set('Choose Gender')
dropdown = OptionMenu(root,tkvar,*choices)
dropdown.place(x=140,y=320)

age = Label(root,text="Date of Birth:",font=(None,12))
age.config(width=0, fg='#FFFFFF', background='#008080')
age.place(x=50,y=390)
dayVar=StringVar(root)
monVar=StringVar(root)
yearVar=StringVar(root)
dayVar.set('Choose Day')
monVar.set('Choose Month')
yearVar.set('Choose Year')
dropdown1 = ttk.Combobox(root,width=19,textvariable=dayVar,state="readonly",values=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'])
dropdown2 = ttk.Combobox(root,width=19,textvariable=monVar,state="readonly",values=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
dropdown3 = ttk.Combobox(root,width=19,textvariable=yearVar,state="readonly",values=['2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2002','2001','2000','1999','1998','1997','1996','1995','1994','1993','1992','1991','1990','1989','1988','1987','1986','1985','1984','1983','1982','1981','1980','1979','1978','1977','1976','1975','1974','1973','1972','1971','1970','1969','1968','1967','1966','1965','1964','1963','1962','1961','1960','1959','1958','1957','1956','1955','1954','1953','1952','1951','1950','1949','1948','1947','1946','1945','1944','1943','1942','1941','1940','1939','1938','1937','1936','1935','1934','1933','1932','1931','1930','1929','1928','1927','1926','1925','1924','1923','1922','1921','1920','1919','1918','1917','1916','1915','1914','1913','1912','1911','1910'])
dropdown1.place(x=175,y=393)
dropdown2.place(x=340,y=393)
dropdown3.place(x=500,y=393)
def pick_date():
    top = Toplevel(root)
    def select():
        dojCal.delete(0,END)
        dojCal.insert(0,cal.selection_get())
        top.destroy()
    cal = Calendar(top,
                   font="Arial 14", selectmode='day',
                   cursor="hand1")
    cal.pack(fill="both", expand=True)
    Button(top,text="OK",command=select).pack()




doj = Label(root,text="Date of Joining:",font=(None,12))
doj.config(width=0, fg='#FFFFFF', background='#008080')
doj.place(x=350,y=320)
dojCal = Entry(root,width=20)
dojCalBtn = Button(root,text="Calendar",width=7,height=0,command=pick_date)
dojCal.place(x=470,y=323)
dojCalBtn.place(x=580,y=320)


des = Label(root, text="Designation:", font=(None,12))
des.config(width=0, fg='#FFFFFF', background='#008080')
desText = Text(root, width=57, height=0)
desText.place(x=175,y=460)

password = Label(root, text="Update Password:", font=(None,12))
password.config(width=0, fg='#FFFFFF', background='#008080')
passwordText = Entry(root, width=54, font=('Verdana',10))
passwordText.config(show="*")
passwordText.place(x=200,y=513)

conpassword = Label(root, text="Confirm Password:", font=(None,12))
conpassword.config(width=0, fg='#FFFFFF', background='#008080')
conpasswordText = Entry(root, width=54, font=('Verdana',10))
conpasswordText.config(show="*")
conpasswordText.place(x=200,y=573)

conpassword.place(x=50,y=570)
password.place(x=50,y=510)
des.place(x=50, y=455)
gender.place(x=50,y=323)
contact.place(x=50, y=250)
address.place(x=50,y=180)
lname.place(x=350,y=120)
fname.place(x=50, y=120)
adddoc.place(x=200,y=0)

def validateContact(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def update_data():
    ans = messagebox.askyesno("Updating data","Are you sure you want to update this record?")
    if(ans == True):
        #try:
            cursor = conn.cursor()
            val2=""
            val3=""
            dob = dayVar.get() + "-" + monVar.get() + "-" + yearVar.get()
            a, b, year = dob.split("-")
            calyear = datetime.datetime.now().year - int(year)
            if(len(passwordText.get())==0):
                if (validateContact(contactText.get("1.0", "end-1c")) == True and len(contactText.get("1.0", "end-1c")) == 11 and (contactText.get("1.0", "end-1c").startswith("03") == True or contactText.get("1.0","end-1c").startswith("0423"))):
                    if(calyear >= 30 and calyear <= 65):
                        if(search.get().startswith("NH-") == True and search.get().endswith("-/NL") == True):
                            sql = "UPDATE BrainTumor.dbo.DoctorInfo SET fname=?,lname=?,addresss=?,contact=?,gender=?,designation=?,dob=?,doj=? WHERE ID=?"
                            val = (fnameText.get("1.0","end-1c"),lnameText.get("1.0","end-1c"),addressText.get("1.0","end-1c"),contactText.get("1.0","end-1c"),str(tkvar.get()),desText.get("1.0","end-1c"),dob,dojCal.get(),search.get())
                            cursor.execute(sql,val)
                            conn.commit()
                            if(messagebox.askokcancel("Update record","Record updated successfully!")):
                                fnameText.delete("1.0", "end-1c")
                                lnameText.delete("1.0", "end-1c")
                                addressText.delete("1.0", "end-1c")
                                desText.delete("1.0", "end-1c")
                                contactText.delete("1.0", "end-1c")
                                passwordText.delete(0, END)
                                conpasswordText.delete(0, END)
                                search.delete(0, END)
                                tkvar.set('Choose Gender')
                                dayVar.set('Choose Day')
                                monVar.set('Choose Month')
                                yearVar.set('Choose Year')
                                search.insert(0, placeholder)
                                dojCal.delete(0, END)
                            else:
                                fnameText.delete("1.0", "end-1c")
                                lnameText.delete("1.0", "end-1c")
                                addressText.delete("1.0", "end-1c")
                                desText.delete("1.0", "end-1c")
                                contactText.delete("1.0", "end-1c")
                                passwordText.delete(0, END)
                                conpasswordText.delete(0, END)
                                search.delete(0, END)
                                tkvar.set('Choose Gender')
                                dayVar.set('Choose Day')
                                monVar.set('Choose Month')
                                yearVar.set('Choose Year')
                                search.insert(0, placeholder)
                                dojCal.delete(0, END)

                        else:
                            messagebox.showinfo("ID value",
                                            "ID must always start with initals \"NH\".\nSee Add Doctor Page in Help menu to continue.")
                    else:
                        messagebox.showinfo("Age value",
                                            "Age selected is not valid")
                else:
                    messagebox.showerror("Error",
                                         "Phone number is either incomplete or incorrect!\nIt must be of format 03xxxxxxxxx or 0423xxxxxxx")
            else:
                if(conpasswordText.get()==passwordText.get()):
                    if (validateContact(contactText.get("1.0", "end-1c")) == True and len(contactText.get("1.0", "end-1c")) == 11 and (contactText.get("1.0", "end-1c").startswith("03") == True or contactText.get("1.0","end-1c").startswith("0423"))):
                        if (calyear >= 30 and calyear <=65):
                            if(search.get().startswith("NH-") == True and search.get().endswith("-/NL") == True):
                                if (len(passwordText.get()) >= 8):
                                    sql = "UPDATE BrainTumor.dbo.DoctorInfo SET fname=?,lname=?,addresss=?,contact=?,gender=?,designation=?,pass=?,dob=?,doj=? WHERE ID=?"
                                    val2 = (fnameText.get("1.0", "end-1c"), lnameText.get("1.0", "end-1c"),
                                    addressText.get("1.0", "end-1c"), contactText.get("1.0", "end-1c"), str(tkvar.get()),
                                    desText.get("1.0", "end-1c"),passwordText.get(),dob,dojCal.get(),search.get())
                                    cursor.execute(sql, val2)
                                    conn.commit()
                                    sql1 = "UPDATE BrainTumor.dbo.Login SET Pass=? WHERE Username=? AND Category=?"
                                    val3 = (passwordText.get(),search.get(),2)
                                    cursor.execute(sql1, val3)
                                    conn.commit()
                                    if (messagebox.askokcancel("Update record", "Record updated successfully!")):
                                        fnameText.delete("1.0", "end-1c")
                                        lnameText.delete("1.0", "end-1c")
                                        addressText.delete("1.0", "end-1c")
                                        desText.delete("1.0", "end-1c")
                                        contactText.delete("1.0", "end-1c")
                                        passwordText.delete(0, END)
                                        conpasswordText.delete(0, END)
                                        search.delete(0, END)
                                        tkvar.set('Choose Gender')
                                        dayVar.set('Choose Day')
                                        monVar.set('Choose Month')
                                        yearVar.set('Choose Year')
                                        search.insert(0, placeholder)
                                        dojCal.delete(0, END)
                                    else:
                                        fnameText.delete("1.0", "end-1c")
                                        lnameText.delete("1.0", "end-1c")
                                        addressText.delete("1.0", "end-1c")
                                        desText.delete("1.0", "end-1c")
                                        contactText.delete("1.0", "end-1c")
                                        passwordText.delete(0, END)
                                        conpasswordText.delete(0, END)
                                        search.delete(0, END)
                                        tkvar.set('Choose Gender')
                                        dayVar.set('Choose Day')
                                        monVar.set('Choose Month')
                                        yearVar.set('Choose Year')
                                        search.insert(0, placeholder)
                                        dojCal.delete(0, END)

                                else:
                                   messagebox.showinfo("Password length",
                                                        "Password must be of 8 or more characters.\nSee Add Doctor Page in Help menu to continue.")
                            else:
                                messagebox.showinfo("ID value",
                                                    "ID must always start with initals \"NH-\" and end with \"-/NL\".\nSee Add Doctor Page in Help menu to continue.")
                        else:
                            messagebox.showinfo("Age value",
                                            "Age selected is not valid")
                    else:
                        messagebox.showerror("Error",
                                             "Phone number is either incomplete or incorrect!\nIt must be of format 03xxxxxxxxx or 0423xxxxxxx")
                else:
                    messagebox.showerror("Error",
                                         "Passwords does not matched!\nPlease enter same password in both fields.")
        #except:
             # messagebox.showerror("Error","Record with input ID is not present!")

def logout():
    res = messagebox.askyesno("Logging Out","Are you sure you want to logout?")
    if(res == True):
        root.destroy()
        call(["python", "Login.py"])
    elif(res == False):
        pass


updateBtn = Button(root, text="Update", height=2,width=20, font=(None,10), command=update_data)

logoutBtn = Button(root, text="Logout", height=2,width=20, font=(None,10), command=logout)

logoutBtn.place(x=400,y=630)
updateBtn.place(x=160,y=630)

def add_data():
    root.destroy()
    call(["python", "DoctorAddAdmin.py"])

def del_data():
    root.destroy()
    call(["python", "DoctorDelAdmin.py"])

def upd_data():
    pass

def abt_us():
    messagebox.showinfo("About Update Doctor Page","*Doctor ID must always starts with initials \"NH\" and end with \"-/NL\".\n*The password must be 8 or more characters long.\n*The minimum criteria of age is 30 years.\n*All fields are mandatory to be filled.")

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
root.title("Update Doctor Information")
root.config(background='#008080')
#root.protocol("WM_DELETE_WINDOW", disable_event)
root.resizable(0,0)
root.mainloop()