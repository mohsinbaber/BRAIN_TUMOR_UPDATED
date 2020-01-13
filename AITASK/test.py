import tkinter as tk
from tkinter import *
def on_keyrelease(event):

    # get text from entry
    value = event.widget.get()
    value = value.strip().lower()

    # get data from test_list
    if value == '':
        data = ""
        listbox.pack_forget()
    else:
        listbox.pack()
        data = []
        for item in test_list:
            if value in item.lower():
                data.append(item)

    # update data in listbox
    listbox_update(data)


def listbox_update(data):
    # delete previous data
    listbox.delete(0, 'end')

    # sorting data
    data = sorted(data, key=str.lower)

    # put new data
    for item in data:
        listbox.insert('end', item)


def on_select(event):
    # display element selected on list
    print('(event) previous:', event.widget.get('active'))
    print('(event)  current:', event.widget.get(event.widget.curselection()))
    print('---')


# --- main ---

test_list = ('apple', 'banana', 'Cranberry', 'dogwood', 'alpha', 'Acorn', 'Anise', 'Strawberry','Pakistan','Pakistan','Pakistan','Pakistan')

root = tk.Tk()

entry = tk.Entry(root)
entry.pack()
entry.insert(END,"Pakistan")
entry.bind('<KeyRelease>', on_keyrelease)

listbox = tk.Listbox(root)
listbox.pack()
listbox.bind('<Double-Button-1>', on_select)
#listbox.bind('<<ListboxSelect>>', on_select)
listbox.pack_forget()
#listbox_update(test_list)
root.geometry("200x200")
root.mainloop()