# import pyperclip
#
# pyperclip.copy("Hello world! This is copied to clipboard via pyperclip")
# pyperclip.paste()

import tkinter as tk

root = tk.Tk()
# keep the window from showing
root.withdraw()

# read the clipboard
m = "hahaha"
c = root.clipboard_get()
root.clipboard_clear()
root.clipboard_append(m)
root.destroy()
print(c)
