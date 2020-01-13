from tkinter import *
from tkinter import messagebox
from PIL import Image
import pyodbc
import io

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')

delete = Tk()

def clear_entry(event, entry):
    if (search.get() == 'Search by ID'):
        entry.delete(0, END)
    else:
        pass

def search_data():
    cursor = conn.cursor()
    val = search.get()
    sql = "SELECT name,ID,dob,gender,contact,PSI FROM BrainTumor.dbo.PatientInfo WHERE ID=?"
    cursor.execute(sql, val)
    result = cursor.fetchall()

    nameText.config(state="normal")
    idText.config(state="normal")
    ageText.config(state="normal")
    genderText.config(state="normal")
    contactText.config(state="normal")

    nameText.delete("1.0", "end-1c")
    idText.delete("1.0", "end-1c")
    ageText.delete("1.0", "end-1c")
    genderText.delete("1.0", "end-1c")
    contactText.delete("1.0", "end-1c")
    img_re2 = Image.open('rec.jpg')
    img_re2 = img_re2.resize((150, 150))
    img_re2.save("abc2.ppm", "ppm")
    init_img = PhotoImage(file='abc2.ppm')
    init_label = Label(image=init_img)
    init_label.image = init_img
    init_label.place(x=700, y=100)

    for row in result:
        nameText.insert(END,row[0])
        idText.insert(END, row[1])
        ageText.insert(END, row[2])
        genderText.insert(END, row[3])
        contactText.insert(END, row[4])
        image = io.BytesIO(row[5])
        img_re = Image.open(image)
        img_re = img_re.resize((150, 150))
        img_re.save("axialSearch.ppm", "ppm")
        imga = PhotoImage(file="axialSearch.ppm")
        imgF = Label(image=imga)
        imgF.image = imga
        imgF.place(x=700, y=100)
        nameText.config(state="disabled")
        idText.config(state="disabled")
        ageText.config(state="disabled")
        genderText.config(state="disabled")
        contactText.config(state="disabled")



search = Entry(delete, width=50, font=('Verdana', 12))
search.place(x=52, y=100)
placeholder = 'Search by ID'
search.insert(0, placeholder)
search.bind("<Button-1>", lambda event: clear_entry(event, search))

srchBtn = Button(delete, text="Search", height=0, width=8, command=search_data)
srchBtn.place(x=540, y=100)

addpat = Label(delete, text="Delete Patient Information", font=(None, 20, 'underline'))
addpat.config(width=0, fg='#FFFFFF', background='#008080')
addpat.place(x=300, y=0)

name = Label(delete, text="Name:", font=(None,15))
name.config(width=0, fg='#FFFFFF', background='#008080')
nameText = Text(delete, width=60, height=0)
nameText.place(x=115,y=175)
name.place(x=50, y=170)

id =  Label(delete, text="Patient ID:", font=(None,15))
id.config(width=0, fg='#FFFFFF', background='#008080')
idText = Text(delete, width=55, height=0)
idText.place(x=150,y=245)
id.place(x=50,y=240)

age =  Label(delete, text="Age:", font=(None,15))
age.config(width=0, fg='#FFFFFF', background='#008080')
ageText = Text(delete, width=20, height=0)
ageText.place(x=150,y=315)
age.place(x=50,y=310)

gender =  Label(delete, text="Gender:", font=(None,15))
gender.config(width=0, fg='#FFFFFF', background='#008080')
genderText = Text(delete, width=20, height=0)
genderText.place(x=430,y=315)
gender.place(x=350,y=310)

contact = Label(delete, text="Contact:", font=(None,15))
contact.config(width=0, fg='#FFFFFF', background='#008080')
contactText = Text(delete, width=55, height=0)
contact.place(x=50,y=380)
contactText.place(x=150,y=385)

img_re2 = Image.open('rec.jpg')
img_re2 = img_re2.resize((150, 150))
img_re2.save("abc2.ppm", "ppm")
init_img = PhotoImage(file='abc2.ppm')
init_label = Label(image=init_img)
init_label.image = init_img
init_label.place(x=700, y=100)
text = Label(delete, text="Axial View", font=(None,15))
text.config(width=0, fg='#FFFFFF', background='#008080')
text.place(x=725, y=270)


def del_data():
    ans = messagebox.askyesno("Deleting data", "Are you sure you want to delete this record?")
    if (ans == True):
        try:
            cursor = conn.cursor()
            val = search.get()
            sql = "DELETE FROM BrainTumor.dbo.PatientInfo WHERE ID=?"
            cursor.execute(sql, val)
            conn.commit()
            messagebox.showinfo("Data deleted", "Data deleted successfully!")

            nameText.config(state="normal")
            idText.config(state="normal")
            ageText.config(state="normal")
            genderText.config(state="normal")
            contactText.config(state="normal")

            nameText.delete("1.0", "end-1c")
            idText.delete("1.0", "end-1c")
            ageText.delete("1.0", "end-1c")
            genderText.delete("1.0", "end-1c")
            contactText.delete("1.0", "end-1c")
            img_re2 = Image.open('rec.jpg')
            img_re2 = img_re2.resize((150, 150))
            img_re2.save("abc2.ppm", "ppm")
            init_img = PhotoImage(file='abc2.ppm')
            init_label = Label(image=init_img)
            init_label.image = init_img
            init_label.place(x=700, y=100)

            nameText.config(state="disabled")
            idText.config(state="disabled")
            ageText.config(state="disabled")
            genderText.config(state="disabled")
            contactText.config(state="disabled")

            search.delete(0, END)
            search.insert(0,placeholder)
        except:
            messagebox.showerror("Error","Record with input ID is not present!")

deleteBtn = Button(delete, text="Delete Data", height=2,width=20, font=(None,10), command=del_data)
deleteBtn.place(x=350,y=580)

delete.geometry("900x700")
delete.title("Add Doctor Information")
delete.config(background='#008080')
delete.resizable(0, 0)
delete.mainloop()
