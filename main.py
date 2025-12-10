import customtkinter as ctk
from src.model import TextClassifier
from PIL import ImageFont
import tkinter.font as tkFont


# -----------------------------
# Load Model
# -----------------------------
clf = TextClassifier(
    model_path="models/svm_model.joblib",
    vectorizer_path="models/tfidf_vectorizer.joblib"
)


# -----------------------------
# Main App (CTk)
# -----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Persian Text Classifier")
app.geometry("600x400")


# -----------------------------
# Load Persian Font (Vazir)
# -----------------------------
FONT_PATH = "assets/fonts/Vazir-FD-WOL.ttf"
FONT_NAME = "Vazir"   # نام داخلی فونت

try:
    # Try to load font using PIL (only to validate the file exists)
    ImageFont.truetype(FONT_PATH, size=14)

    # Register the font in Tk
    app.tk.call("font", "create", FONT_NAME, "-family", FONT_NAME, "-size", 14)

    # Create CTk usable font
    default_font = ctk.CTkFont(family=FONT_NAME, size=14)
    title_font = ctk.CTkFont(family=FONT_NAME, size=20)

    print("Persian font loaded successfully.")

except Exception as e:
    print("Font load failed:", e)
    default_font = ctk.CTkFont(size=14)
    title_font = ctk.CTkFont(size=20)


# -----------------------------
# UI Components
# -----------------------------

title_label = ctk.CTkLabel(
    app,
    text="سامانه تشخیص موضوع متن فارسی",
    font=title_font
)
title_label.pack(pady=20)


input_box = ctk.CTkTextbox(
    app,
    width=500,
    height=120,
    font=default_font
)
input_box.pack(pady=10)


def classify_text():
    text = input_box.get("1.0", "end").strip()
    if not text:
        output_label.configure(text="لطفاً یک متن وارد کنید!", text_color="red")
        return
    
    result = clf.predict(text)
    
    output_label.configure(
        text=f"🎯 موضوع تشخیص داده شده:  {result}",
        text_color="lightgreen"
    )


classify_button = ctk.CTkButton(
    app,
    text="تشخیص موضوع",
    command=classify_text,
    font=default_font
)
classify_button.pack(pady=15)


output_label = ctk.CTkLabel(
    app,
    text="",
    font=default_font
)
output_label.pack(pady=10)


app.mainloop()
