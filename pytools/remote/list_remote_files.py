import pysftp
import tkinter as tk
from tkinter import simpledialog
import argparse

DIR_PATH = '/data2/home/zhouxiangyong/Workspace/Dev/AortaSlice/data/aorta_extract_data1_fxc'

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default=DIR_PATH, type=str, help="src image dir")
args = parser.parse_args()
dir_path = args.dir

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

with pysftp.Connection('s108', port=10822, username='zhouxiangyong', password='zxy201811', cnopts=cnopts) as sftp:
    sftp.cwd(dir_path)
    out = sftp.listdir()
    print(out)


class CustomDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, text=None):
        self.data = text
        simpledialog.Dialog.__init__(self, parent, title=title)

    def body(self, parent):
        self.text = tk.Text(self, width=80, height=4)
        self.text.pack(fill="both", expand=True)
        self.text.insert("1.0", self.data)
        return self.text


def show_dialog():
    fromonk_text = "this is an example"
    CustomDialog(root, title="Example", text=fromonk_text)


root = tk.Tk()
top = tk.Toplevel()
top.title('TkFloat')
button = tk.Button(top, text="Show", command=show_dialog)
button.pack(padx=20, pady=20)
button1 = tk.Button(top, text="Close", command=top.destroy)
button1.pack(padx=20, pady=20)
w = tk.Canvas(top, width=200, height=360)
w.pack()
var1 = tk.IntVar()
tk.Checkbutton(w, text='male', variable=var1).grid(row=0, sticky=tk.W)
var2 = tk.IntVar()
tk.Checkbutton(w, text='female', variable=var2).grid(row=1, sticky=tk.W)

tk.Label(w, text='First Name').grid(row=2)
tk.Label(w, text='Last Name').grid(row=3)
e1 = tk.Entry(w)
e2 = tk.Entry(w)
e1.grid(row=2, column=1)
e2.grid(row=3, column=1)
ourMessage = 'This is our Message'
messageVar = tk.Message(top, text=ourMessage)
messageVar.config(bg='lightgreen')
messageVar.pack()
root.mainloop()
