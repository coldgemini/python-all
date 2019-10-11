import tkinter
from tkinter import messagebox

msg = tkinter.Tk()
msg.withdraw()
messagebox.showinfo("Title", "a Tk MessageBox\nline2")
# messagebox.showinfo("Title", "a Tk MessageBox2")
msg.destroy()

