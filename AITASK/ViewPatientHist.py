from tkinter import *
from tkinter.ttk import *
import pyodbc


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')

class App(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.CreateUI()
        self.LoadTable()
        self.grid(sticky = (N,S,W,E))
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)

    def CreateUI(self):
        tv = Treeview(self)
        tv['columns'] = ('name', 'dob','gender','contact','dov','cancer','stage','pres')
        tv.heading("#0", text='Patient ID')
        tv.column("#0", anchor='center',width=100)
        tv.heading('name', text='Name')
        tv.column('name', anchor='center', width=80)
        tv.heading('dob', text='Date Of Birth')
        tv.column('dob', anchor='center', width=100)
        tv.heading('gender', text='Gender')
        tv.column('gender', anchor='center', width=80)
        tv.heading('contact', text='Contact')
        tv.column('contact', anchor='center', width=100)
        tv.heading('dov', text='Date of Appointment')
        tv.column('dov', anchor='center', width=120)
        tv.heading('cancer', text='Cancer')
        tv.column('cancer', anchor='center', width=140)
        tv.heading('stage', text='Stage')
        tv.column('stage', anchor='center', width=100)
        tv.heading('pres', text='Prescription')
        tv.column('pres', anchor='center', width=100)
        tv.grid(sticky = (N,S,W,E))
        self.treeview = tv
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)

    def LoadTable(self):
        cursor = conn.cursor()
        sql = "SELECT * FROM BrainTumor.dbo.PatientHist"
        cursor.execute(sql)
        result = cursor.fetchall()
        count = 1
        for row in result:
            count = count + 1

        i = 0

        ID = [col[0] for col in result]
        name = [col[1] for col in result]
        age = [col[2] for col in result]
        gender = [col[3] for col in result]
        contact = [col[4] for col in result]
        dov = [col[5] for col in result]
        cancer = [col[6] for col in result]
        stage = [col[8] for col in result]
        pres = [col[9] for col in result]
        while (i < count - 1):
            self.treeview.insert('', 'end', text=str(ID[i]), values=(str(name[i]),age[i],gender[i],contact[i],dov[i],cancer[i],stage[i],pres[i]))
            self.treeview.bind("<Double-1>", self.OnDoubleClick)

            i = i+1

    def OnDoubleClick(self, event):
        pat_id = self.treeview.selection()[0]
        item = self.treeview.selection()
        root = Tk()
        adddoc = Label(root, text="Patient Information", font=(None, 20, 'underline'))
        adddoc.config(width=0, foreground='#000000')

        lbId = Label(root,text="Patient ID", font=(None, 15))
        lbId.place(x=30, y=90)
        lb = Label(root, text=self.treeview.item(pat_id, "text"), font=(None, 15))
        lb.place(x=170,y=90)

        name = Label(root, text="Name:", font=(None, 15))
        name.place(x=30, y=130)

        age = Label(root, text="Date of Birth:", font=(None, 15))
        age.place(x=30, y=170)

        gender = Label(root, text="Gender:", font=(None, 15))
        gender.place(x=30, y=210)

        contact = Label(root, text="Contact:", font=(None, 15))
        contact.place(x=30, y=250)

        dov = Label(root, text="Date of Appointment:", font=(None, 15))
        dov.place(x=30, y=290)

        cancer = Label(root, text="Cancer Type:", font=(None, 15))
        cancer.place(x=30, y=330)

        stage = Label(root, text="Stage:", font=(None, 15))
        stage.place(x=30, y=370)

        pres = Label(root, text="Prescription:", font=(None, 15))
        pres.place(x=30, y=410)


        for i in item:
            j = 0
            y_axis=0
            while j < len(i)+4:
                lb2 = Label(root,text=self.treeview.item(i, "values")[j], font=(None, 15))
                lb2.place(x=170,y=130+y_axis)
                y_axis = y_axis+40
                j = j+1
        adddoc.place(x=200, y=0)
        root.title("Patient Information")
        printBtn = Button(root,text="Print",width=20)
        printBtn.place(x=200,y=500)
        root.geometry("600x600")
        root.mainloop()

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == '__main__':
    main()