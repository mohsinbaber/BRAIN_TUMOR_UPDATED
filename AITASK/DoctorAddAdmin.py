from tkinter import *
import pyodbc
from tkinter import messagebox
from subprocess import call
from tkinter import ttk
from tkcalendar import DateEntry,Calendar
import datetime

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')
root = Tk()

menubar = Menu(root)
root.config(menu=menubar)

adddoc = Label(root, text="Add Doctor Information", font=(None,20,'underline'))
#adddoc.config(width=0, fg='#FFFFFF', background='#008080')

idd = Label(root, text="ID/Username:", font=(None,12))
idd.place(x=50,y=70)
#idd.config(width=0, fg='#FFFFFF', background='#008080')
iddText = Text(root, width=54, height=0)
iddText.place(x=195,y=73)
iddText.config(state="disabled")

def get_id():
    que = "SELECT MAX(CAST(SUBSTRING(ID, 4, len(ID)-3) AS int)) FROM BrainTumor.dbo.DoctorInfo"
    cursor1 = conn.cursor()
    cursor1.execute(que)
    result = cursor1.fetchall()
    cou = 0
    idd=""
    for rows in result:
        idd = rows[cou]
    if(idd == None):
        idAdd = "1001"
    else:
        idFinal = int(idd)+1
        idAdd = "100"+str(idFinal)
    return idAdd

iddText.config(state="normal")
iddText.insert(END,get_id())
iddText.config(state="disabled")

fname = Label(root, text="First Name:", font=(None,12))
#fname.config(width=0, fg='#FFFFFF', background='#008080')
fnameText = Text(root, width=25, height=0)
fnameText.place(x=135,y=132)

lname = Label(root, text="Last Name:", font=(None,12))
#lname.config(width=0, fg='#FFFFFF', background='#008080')
lnameText = Text(root, width=25, height=0)
lnameText.place(x=435,y=132)

address = Label(root, text="Address:", font=(None,12))
#address.config(width=0, fg='#FFFFFF', background='#008080')
addressText = Text(root, width=22, height=0)
addressText.place(x=195,y=203)

contact = Label(root, text="Contact Number:", font=(None,12))
#contact.config(width=0, fg='#FFFFFF', background='#008080')
contactText = Text(root, width=54, height=0)
contactText.place(x=195,y=263)

gender = Label(root, text="Gender:", font=(None,12))
choices = {'Male', 'Female','Others'}
tkvar = StringVar(root)
tkvar.set('Choose Gender')
dropdown = ttk.Combobox(root,textvariable=tkvar,state="readonly",values=['Male', 'Female','Others'])
dropdown.place(x=195,y=320)

age = Label(root,text="Date of Birth:",font=(None,12))
age.place(x=50,y=380)
dayVar=StringVar(root)
monVar=StringVar(root)
yearVar=StringVar(root)
dayVar.set('Choose Day')
monVar.set('Choose Month')
yearVar.set('Choose Year')
dropdown1 = ttk.Combobox(root,width=19,textvariable=dayVar,state="readonly",values=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'])
dropdown2 = ttk.Combobox(root,width=19,textvariable=monVar,state="readonly",values=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
dropdown3 = ttk.Combobox(root,width=19,textvariable=yearVar,state="readonly",values=['2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2002','2001','2000','1999','1998','1997','1996','1995','1994','1993','1992','1991','1990','1989','1988','1987','1986','1985','1984','1983','1982','1981','1980','1979','1978','1977','1976','1975','1974','1973','1972','1971','1970','1969','1968','1967','1966','1965','1964','1963','1962','1961','1960','1959','1958','1957','1956','1955','1954','1953','1952','1951','1950','1949','1948','1947','1946','1945','1944','1943','1942','1941','1940','1939','1938','1937','1936','1935','1934','1933','1932','1931','1930','1929','1928','1927','1926','1925','1924','1923','1922','1921','1920','1919','1918','1917','1916','1915','1914','1913','1912','1911','1910'])
dropdown1.place(x=195,y=383)
dropdown2.place(x=350,y=383)
dropdown3.place(x=500,y=383)

doj = Label(root,text="Date of Joining:",font=(None,12))
doj.place(x=350,y=320)
cal = DateEntry(root, width=12, background='darkblue',
                    foreground='white', borderwidth=2)
cal.place(x=465, y=323)

des = Label(root, text="Designation:", font=(None,12))
desText = Text(root, width=54, height=0)
desText.place(x=195,y=453)

password = Label(root, text="Set Password:", font=(None,12))
passwordText = Entry(root, width=54, font=('Verdana',10))
passwordText.config(show="*")
passwordText.place(x=195,y=523)

conpassword = Label(root, text="Confirm Password:", font=(None,12))
conpasswordText = Entry(root, width=54, font=('Verdana',10))
conpasswordText.config(show="*")
conpasswordText.place(x=195,y=593)

conpassword.place(x=50,y=590)
password.place(x=50,y=520)
des.place(x=50, y=450)
gender.place(x=50,y=320)
contact.place(x=50, y=260)
address.place(x=50,y=200)
lname.place(x=350,y=130)
fname.place(x=50, y=130)
adddoc.place(x=220,y=0)

def clear_entry(event, entry):
    if(entry.get() == 'Choose City'):
        entry.delete(0, END)
    else:
        pass
cityArr = ('Karachi','Lahore','Faisalabad','Serai','Rawalpindi','Multan','Gujranwala','Hyderabad City','Peshawar','Abbottabad','Islamabad','Quetta','Bannu','Bahawalpur','Sargodha','Sialkot City','Sukkur','Larkana','Sheikhupura','Mirpur Khas','Rahimyar Khan','Kohat','Jhang Sadr','Gujrat','Bardar','Kasur','Dera Ghazi Khan','Masiwala','Nawabshah','Okara','Gilgit','Chiniot','Sadiqabad','Turbat','Dera Ismail Khan','Chaman','Zhob','Mehra','Parachinar','Gwadar','Kundian','Shahdad Kot','Haripur','Matiari','Dera Allahyar','Lodhran','Batgram','Thatta','Bagh','Badin','Mansehra','Ziarat','Muzaffargarh','Tando Allahyar','Dera Murad Jamali','Karak','Mardan','Uthal','Nankana Sahib','Barkhan','Hafizabad','Kotli','Loralai','Dera Bugti','Jhang City','Sahiwal','Sanghar','Pakpattan','Chakwal','Khushab','Ghotki','Kohlu','Khuzdar','Awaran','Nowshera','Charsadda','Qila Abdullah','Bahawalnagar','Dadu','Aliabad','Lakki Marwat','Chilas','Pishin','Tank','Chitral','Qila Saifullah','Shikarpur','Panjgur','Mastung','Kalat','Gandava','Khanewal','Narowal','Khairpur','Malakand','Vihari','Saidu Sharif','Jhelum','Mandi Bahauddin','Bhakkar','Toba Tek Singh','Jamshoro','Kharan','Umarkot','Hangu','Timargara','Gakuch','Jacobabad','Alpurai','Mianwali','Musa Khel Bazar','Naushahro Firoz','New Mirpur','Daggar','Eidgah','Sibi','Dalbandīn','Rajanpur','Leiah','Upper Dir','Tando Muhammad Khan','Attock City','Rawala Kot','Swabi','Kandhkot','Dasu','Athmuqam')

def on_keyreleaseCity(event):
    value = event.widget.get()
    value = value.strip().lower()

    # get data from test_list
    if value == '':
        data = ""
        cityList.place_forget()
    else:
        cityList.place(x=380,y=223)
        data = []
        for item in cityArr:
            if value in item.lower():
                data.append(item)

    # update data in listbox
    listbox_updateCity(data)

def listbox_updateCity(data):
    # delete previous data
    cityList.delete(0, 'end')

    # sorting data
    data = sorted(data, key=str.lower)

    # put new data
    for item in data:
        cityList.insert('end', item)

def on_selectCity(event):
    entry.delete(0, 'end')
    print('(event) previous:', event.widget.get('active'))
    entry.insert('end',event.widget.get(event.widget.curselection()))
    cityList.place_forget()

cityList = Listbox(root)
cityList.place(x=380,y=223)
cityList.place_forget()
entry = Entry(root)
placeholder='Choose City'
entry.insert(0,placeholder)
entry.bind("<Button-1>", lambda event: clear_entry(event,entry))
entry.bind('<KeyRelease>', on_keyreleaseCity)
entry.place(x=380,y=203)
cityList.bind('<Double-Button-1>', on_selectCity)


def clear_entryCountry(event, entry2):
    if(entry2.get() == 'Choose Country'):
        entry2.delete(0, END)
    else:
        pass

def on_keyreleaseCountry(event):
    value = event.widget.get()
    value = value.strip().lower()

    # get data from test_list
    if value == '':
        data = ""
        countryList.place_forget()
    else:
        countryList.place(x=510,y=223)
        data = []
        for item in countryArr:
            if value in item.lower():
                data.append(item)

    # update data in listbox
    listbox_updateCountry(data)

def listbox_updateCountry(data):
    # delete previous data
    countryList.delete(0, 'end')

    # sorting data
    data = sorted(data, key=str.lower)

    # put new data
    for item in data:
        countryList.insert('end', item)

def on_selectCountry(event):
    entry2.delete(0, 'end')
    print('(event) previous:', event.widget.get('active'))
    entry2.insert('end',event.widget.get(event.widget.curselection()))
    countryList.place_forget()

countryArr = ("Andorra",
    "United Arab Emirates",
    "Afghanistan",
    "Antigua And Barbuda",
    "Anguilla",
    "Albania",
    "Armenia",
    "Angola",
    "Argentina",
    "American Samoa",
    "Austria",
    "Australia",
    "Aruba",
    "Azerbaijan",
    "Bosnia And Herzegovina",
    "Barbados",
    "Bangladesh",
    "Belgium",
    "Burkina Faso",
    "Bulgaria",
    "Bahrain",
    "Burundi",
    "Benin",
    "Saint Barthelemy",
    "Bermuda", "Brunei",
    "Bolivia",
    "Brazil",
    "Bahamas, The",
    "Bhutan",
    "Botswana",
    "Belarus",
    "Belize",
    "Canada",
    "Cocos (Keeling) Islands",
    "Congo (Kinshasa)",
    "Central African Republic",
    "Congo (Brazzaville)",
    "Switzerland",
    "Côte D’Ivoire (Ivory Coast)",
    "Cook Islands",
    "Chile",
    "Cameroon",
    "China",
    "Colombia",
    "Costa Rica",
    "Cuba",
    "Cabo Verde",
    "Curaçao",
    "Christmas Island",
    "Cyprus",
    "Czechia",
    "Germany",
    "Djibouti",
    "Denmark",
    "Dominica",
    "Dominican Republic",
    "Algeria",
    "Ecuador",
    "Estonia",
    "Egypt",
    "Western Sahara",
    "Eritrea",
    "Spain",
    "Ethiopia",
    "Finland",
    "Fiji",
    "Falkland Islands (Islas Malvinas)",
    "Micronesia, Federated States Of",
    "Faroe Islands",
    "France",
    "Gabon",
    "United Kingdom",
    "Grenada",
    "Georgia",
    "French Guiana",
    "Guernsey",
    "Ghana",
    "Gibraltar",
    "Greenland",
    "Gambia, The",
    "Guinea",
    "Guadeloupe",
    "Equatorial Guinea",
    "Greece",
    "South Georgia And South Sandwich Islands",
    "Guatemala",
     "Guam",
     "Guinea-Bissau",
    "Guyana",
    "Hong Kong",
    "Honduras",
    "Croatia",
    "Haiti",
    "Hungary",
    "Indonesia",
    "Ireland",
    "Israel",
    "Isle Of Man",
    "India",
    "British Indian Ocean Territory",
    "Iraq",
    "Iran",
    "Iceland",
    "Italy",
    "Jersey",
    "Jamaica",
    "Jordan",
    "Japan",
    "Kenya",
    "Kyrgyzstan",
    "Cambodia",
    "Kiribati",
    "Comoros",
    "Saint Kitts And Nevis",
    "Korea, North",
    "Korea, South",
    "Kuwait",
    "Cayman Islands",
    "Kazakhstan",
    "Laos",
    "Lebanon",
    "Saint Lucia",
    "Liechtenstein",
    "Sri Lanka",
    "Liberia",
    "Lesotho",
    "Lithuania",
    "Luxembourg",
    "Latvia",
    "Libya",
    "Morocco",
    "Monaco",
    "Moldova",
    "Montenegro",
    "Saint Martin",
    "Madagascar",
    "Marshall Islands",
    "Macedonia",
    "Mali",
    "Burma",
    "Mongolia",
    "Macau",
    "Northern Mariana Islands",
    "Martinique",
    "Mauritania",
    "Montserrat",
    "Malta",
    "Mauritius",
    "Maldives",
    "Malawi",
    "Mexico",
    "Malaysia",
    "Mozambique",
    "Namibia", "New Caledonia",
    "Niger",
    "Norfolk Island",
    "Nigeria",
    "Nicaragua",
    "Netherlands",
    "Norway",
    "Nepal",
    "Nauru",
    "Niue",
    "New Zealand",
    "Oman",
    "Panama",
    "Peru",
    "French Polynesia",
    "Papua New Guinea",
    "Philippines",
    "Pakistan",
    "Poland",
    "Saint Pierre And Miquelon",
    "Pitcairn Islands",
    "Puerto Rico",
    "Portugal",
    "Palau",
    "Paraguay",
    "Qatar",
    "Akrotiri",
    "Reunion",
    "Romania",
    "Serbia",
    "Russia",
    "Rwanda",
    "Saudi Arabia",
    "Solomon Islands",
    "Seychelles",
    "Sudan", "Sweden",
    "Singapore",
    "Saint Helena, Ascension, And Tristan Da Cunha",
    "Slovenia", "Slovakia",
    "Sierra Leone",
    "San Marino",
    "Senegal",
    "Somalia",
    "Suriname",
    "South Sudan",
    "Sao Tome And Principe",
    "El Salvador",
    "Sint Maarten",
    "Syria",
    "Swaziland", "Turks And Caicos Islands",
    "Chad",
    "French Southern And Antarctic Lands",
    "Togo",
    "Thailand",
    "Tajikistan",
    "Tokelau",
    "Timor-Leste",
    "Turkmenistan",
    "Tunisia",
    "Tonga",
    "Turkey",
    "Trinidad And Tobago",
    "Tuvalu",
    "Taiwan",
    "Tanzania",
    "Ukraine",
    "Uganda",
    "United States",
    "Uruguay",
    "Uzbekistan",
    "Saint Vincent And The Grenadines",
    "Venezuela",
    "Virgin Islands, British",
    "Virgin Islands, U.S.",
    "Vietnam",
    "Vanuatu",
    "Wallis And Futuna",
    "Samoa",
    "Dhekelia",
    "Gaza Strip",
    "Kosovo",
    "Paracel Islands",
    "Svalbard",
    "Spratly Islands",
    "West Bank",
    "Yemen",
    "Mayotte",
    "South Africa",
    "Zambia",
    "Zimbabwe")


countryList = Listbox(root)
countryList.place(x=510,y=223)
countryList.place_forget()
entry2 = Entry(root)
entry2.insert(END,"Choose Country")
entry2.bind("<Button-1>", lambda event: clear_entryCountry(event,entry2))
entry2.bind('<KeyRelease>', on_keyreleaseCountry)
entry2.place(x=510,y=203)
countryList.bind('<Double-Button-1>', on_selectCountry)


def validateContact(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def save_data():
    ans = messagebox.askyesno("Save Information","Do you wish to continue?")
    if(ans == True):
        #que = "SELECT MAX(CAST(SUBSTRING(ID, 4, len(ID)-3) AS int)) FROM BrainTumor.dbo.DoctorInfo"
        que = 'SELECT MAX(CAST(SUBSTRING(ID, 4, len(ID)-3) AS int)) FROM BrainTumor.dbo.DoctorInfo'
        cursor1 = conn.cursor()
        cursor1.execute(que)
        result = cursor1.fetchall()
        cou = 0

        idd=""
        for rows in result:
            idd = rows[cou]

        if(idd == None):
            idAdd = "1001"
        else:
            idFinal = int(idd)+1
            idAdd = "100"+str(idFinal)
        dob = dayVar.get()+"-"+monVar.get()+"-"+yearVar.get()
        a,b,year = dob.split("-")
        calyear = datetime.datetime.now().year - int(year)

        if(len(fnameText.get("1.0", "end-1c")) !=0 and len(lnameText.get("1.0", "end-1c")) !=0 and len(addressText.get("1.0", "end-1c")) !=0 and (entry.get() != "Choose City") and (entry2.get() !="Choose Country") and len(contactText.get("1.0", "end-1c")) !=0 and tkvar.get() != 'Choose Gender' and len(str(cal.get_date())) !=0 and dayVar.get()!='Choose Day' and monVar.get()!='Choose Month' and yearVar.get()!='Choose Year' and len(desText.get("1.0", "end-1c")) !=0 and len(passwordText.get()) !=0 and len(conpasswordText.get()) !=0):
            if(len(contactText.get("1.0", "end-1c")) == 11 and (contactText.get("1.0", "end-1c").startswith("03") == True or contactText.get("1.0", "end-1c").startswith("0423"))):
                if(calyear >=30 and calyear<=65):
                    if(len(passwordText.get()) >=8):
                        if(conpasswordText.get() == passwordText.get()):
                            if(validateContact(contactText.get("1.0", "end-1c")) == True):
                                temp = str(addressText.get("1.0", "end-1c"))+","+str(entry.get())+","+str(entry2.get())
                                cursor = conn.cursor()
                                sql = "INSERT INTO BrainTumor.dbo.DoctorInfo (ID,fname,lname,addresss,contact,gender,designation,pass,dob,doj)VALUES (?,?,?,?,?,?,?,?,?,?)"
                                val = (idAdd,fnameText.get("1.0", "end-1c"),lnameText.get("1.0", "end-1c"),temp,contactText.get("1.0", "end-1c"),tkvar.get(),desText.get("1.0", "end-1c"),passwordText.get(),dob,str(cal.get_date()))
                                cursor.execute(sql,val)
                                conn.commit()

                                sql2 = "INSERT INTO BrainTumor.dbo.Login (Username,Pass,Category)VALUES (?,?,?)"
                                val2 = (idAdd,passwordText.get(),2)
                                cursor2 = conn.cursor()
                                cursor2.execute(sql2,val2)
                                conn.commit()

                                if(messagebox.askokcancel("New Record","Record entered successfully!")):
                                    fnameText.delete("1.0", "end-1c")
                                    lnameText.delete("1.0", "end-1c")
                                    addressText.delete("1.0", "end-1c")
                                    desText.delete("1.0", "end-1c")
                                    contactText.delete("1.0", "end-1c")
                                    passwordText.delete(0, END)
                                    conpasswordText.delete(0, END)
                                    tkvar.set('Choose Gender')
                                    dayVar.set('Choose Day')
                                    monVar.set('Choose Month')
                                    yearVar.set('Choose Year')
                                    cal.delete(0, END)
                                    iddText.config(state="normal")
                                    iddText.delete("1.0", "end-1c")
                                    iddText.insert(END, get_id())
                                    iddText.config(state="disabled")
                                    entry.delete(0,END)
                                    entry2.delete(0, END)
                                    entry2.insert(END, "Choose Country")
                                    entry.insert(END, "Choose City")

                                else:
                                    fnameText.delete("1.0", "end-1c")
                                    lnameText.delete("1.0", "end-1c")
                                    addressText.delete("1.0", "end-1c")
                                    desText.delete("1.0", "end-1c")
                                    contactText.delete("1.0", "end-1c")
                                    passwordText.delete(0, END)
                                    conpasswordText.delete(0, END)
                                    tkvar.set('Choose Gender')
                                    dayVar.set('Choose Day')
                                    monVar.set('Choose Month')
                                    yearVar.set('Choose Year')
                                    cal.delete(0, END)
                                    iddText.config(state="normal")
                                    iddText.delete("1.0", "end-1c")
                                    iddText.insert(END, get_id())
                                    iddText.config(state="disabled")
                                    entry.delete(0, END)
                                    entry2.delete(0, END)
                                    entry2.insert(END, "Choose Country")
                                    entry.insert(END, "Choose City")

                            else:
                                messagebox.showerror("Error", "Contact number must be numeric not characters")
                        else:
                            messagebox.showerror("Error", "Wrong password is entered in Confirm Password!")
                    else:
                        messagebox.showerror("Error", "Password must be of 8 or more characters!")
                else:
                    messagebox.showerror("Error","Age selected is not valid")
            else:
                messagebox.showerror("Error","Phone number is either incomplete or incorrect!!\nIt must be of format 03xxxxxxxxx or 0423xxxxxxx")
        else:
            messagebox.showerror("Error", "One or more fields are empty!")

def logout():
    res = messagebox.askyesno("Logging Out","Are you sure you want to logout?")
    if(res == True):
        root.destroy()
        call(["python", "Login.py"])
    elif(res == False):
        pass

def on_enter(e):
    saveBtn['background'] = '#F0F0F0'

def on_leave(e):
    saveBtn['background'] = '#CBCBCB'

saveBtn = Button(root, text="Save", height=2,width=20, font=(None,10),background="#CBCBCB", command=save_data)
saveBtn.bind("<Enter>",on_enter)
saveBtn.bind("<Leave>",on_leave)

def on_enter(e):
    logoutBtn['background'] = '#F23838'

def on_leave(e):
    logoutBtn['background'] = '#CBCBCB'

logoutBtn = Button(root, text="Logout", height=2,width=20, font=(None,10),background="#CBCBCB", command=logout)
logoutBtn.bind("<Enter>",on_enter)
logoutBtn.bind("<Leave>",on_leave)
logoutBtn.place(x=400,y=630)


saveBtn.place(x=160,y=630)
def del_data():
    root.destroy()
    call(["python", "DoctorDelAdmin.py"])

def upd_data():
    root.destroy()
    call(["python", "DoctorUpdAdmin.py"])

def add_data():
    pass

def abt_us():
    messagebox.showinfo("About Us", "This application segments and detect brain tumors in an MRI Image.")

def abt_dp():
    messagebox.showinfo("About Add Doctor Page","*The password must be 8 or more characters long.\n*The minimum criteria of age is 30 years.\n*All fields are mandatory to be filled.")

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
submenu2.add_command(label="About Add Doctor Page",command=abt_dp)

def disable_event():
    if(messagebox.askyesno("Logging Out","Do you wish to log out?")):
        root.destroy()
        call(["python", "Login.py"])
    else:
        pass

root.geometry("710x700")
root.title("Add Doctor Information")
root.resizable(0,0)
root.mainloop()