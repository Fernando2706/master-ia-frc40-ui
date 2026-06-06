import tkinter as tk


def log_safe(text_widget: tk.Text, message: str) -> None:
    text_widget.configure(state="normal")
    text_widget.insert("end", message + "\n")
    text_widget.see("end")
    text_widget.configure(state="disabled")

