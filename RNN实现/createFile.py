## create a class that can conver the string
## to the PDF file
from fpdf import FPDF
class PDFConverter:
    def __init__(self, string):
        self.string = string
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(200, 10, txt=self.string, ln=1, align="C")
        self.pdf.output("test.pdf")
        self.pdf.close()

# create a file write
class TextFileWriter:
    def __init__(self, string, encoding="UTF-8"):
        self.string = string
        self.file = open("test.txt", "w", encoding=encoding)
        self.file.write(self.string)
        self.file.close()



if __name__ == "__main__":
    # test the class
    string = "This is a test string"
    pdf = PDFConverter(string)
    print("Done")