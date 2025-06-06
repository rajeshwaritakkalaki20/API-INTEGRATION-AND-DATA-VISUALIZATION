from PyQt5.QtWidgets import QPushButton, QLineEdit, QApplication, QFormLayout, QWidget, QTextEdit, QSpinBox # type: ignore
from dataclasses import field
class Window(QWidget):

    def _init_(self):
        super()._init_()

        self.name = QLineEdit()
        self.program_type = QLineEdit()
        self.product_code = QLineEdit()
        self.customer = QLineEdit()
        self.vendor = QLineEdit()
        self.n_errors = QSpinBox()
        self.n_errors.setRange(0, 1000)
        self.comments = QTextEdit()

        self.generate_btn = QPushButton("Generate PDF")

        layout = QFormLayout()
        layout.addRow("Name", self.name)
        layout.addRow("Program Type", self.program_type)
        layout.addRow("Product Code", self.product_code)
        layout.addRow("Customer", self.customer)
        layout.addRow("Vendor", self.vendor)
        layout.addRow("No. of Errors", self.n_errors)

        layout.addRow("Comments", self.comments)
        layout.addRow(self.generate_btn)

        self.setLayout(layout)


app = QApplication([])
w = Window()
w.show()
app.exec()
outfile = "result.pdf"

template = PdfReader("template.pdf", decompress=False).pages[0]
template_obj = pagexobj(template)

canvas = Canvas(outfile)

xobj_name = makerl(canvas, template_obj)
canvas.doForm(xobj_name)

ystart = 443


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    error = pyqtSignal(str)
    file_saved_as = pyqtSignal(str)


class Generator(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handle worker thread setup, signals
    and wrap-up.

    :param data: The data to add to the PDF for generating.
    """

    def _init_(self, data):
        super()._init_()
        self.data = data
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            outfile = "result.pdf"

            template = PdfReader("template.pdf", decompress=False).pages[0]
            template_obj = pagexobj(template)

            canvas = Canvas(outfile)

            xobj_name = makerl(canvas, template_obj)
            canvas.doForm(xobj_name)

            ystart = 443

            # Prepared by
            canvas.drawString(170, ystart, self.data['name'])

            canvas.save()

        except Exception as e:
            self.signals.error.emit(str(e))
            return

        self.signals.file_saved_as.emit(outfile)
class Window(QWidget):

    def _init_(self):
        super()._init_()

        self.threadpool = QThreadPool()
def generate(self):
    self.generate_btn.setDisabled(True)
    data = {
        'name': self.name.text(),
        'program_type': self.program_type.text(),
        'product_code': self.product_code.text(),
        'customer': self.customer.text(),
        'vendor': self.vendor.text(),
        'n_errors': str(self.n_errors.value()),
        'comments': self.comments.toPlainText()
    }
    g = Generator(data)
    g.signals.file_saved_as.connect(self.generated)
    g.signals.error.connect(print)  # Print errors to console.
    self.threadpool.start(g)

def generated(self, outfile):
    self.generate_btn.pressed.connect(self.generate)
ystart = 443

# Prepared by
canvas.drawString(170, ystart, self.data['name']) # type: ignore

# Date: Todays date
today = datetime.today()
canvas.drawString(410, ystart, today.strftime('%F'))

# Device/Program Type
canvas.drawString(230, ystart-28, self.data['program_type']) # type: ignore

# Product code
canvas.drawString(175, ystart-(2*28), self.data['product_code']) # type: ignore

# Customer
canvas.drawString(315, ystart-(2*28), self.data['customer']) # type: ignore

# Vendor
canvas.drawString(145, ystart-(3*28), self.data['vendor']) # type: ignore

ystart = 250

# Program Language
canvas.drawString(210, ystart, "Python")

canvas.drawString(430, ystart, Self.data['n_errors']) # type: ignore
field.setMaxLength(25)
import textwrap
comments = comments.replace('\n', ' ') # type: ignore
lines = textwrap.wrap(comments, width=80)
import textwrap
comments = comments.replace('\n', ' ')
lines = textwrap.wrap(comments, width=65) # 45
first_line = lines[0]
remainder = ' '.join(lines[1:])

lines = textwrap.wrap(remainder, 75) # 55
lines = lines[:4]  # max lines, not including the first.
comments = self.data['comments'].replace('\n', ' ') # type: ignore
if comments:
    lines = textwrap.wrap(comments, width=65) # 45
    first_line = lines[0]
    remainder = ' '.join(lines[1:])

    lines = textwrap.wrap(remainder, 75) # 55
    lines = lines[:4]  # max lines, not including the first.

    canvas.drawString(155, 223, first_line)
    for n, l in enumerate(lines, 1):
        canvas.drawString(80, 223 - (n*28), l)

    def generated(self, outfile):
        self.generate_btn.setDisabled(False)
        try:
            os.startfile(outfile)
        except Exception:
            # If startfile not available, show dialog.
            QMessageBox.information(self, "Finished", "PDF has been generated")
from PyQt5.QtWidgets import QPushButton, QLineEdit, QApplication, QFormLayout, QWidget, QTextEdit, QMessageBox, QSpinBox # type: ignore
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot # type: ignore

from reportlab.pdfgen.canvas import Canvas # type: ignore

import os

import textwrap
from datetime import datetime

from pdfrw import PdfReader # type: ignore
from pdfrw.buildxobj import pagexobj # type: ignore
from pdfrw.toreportlab import makerl # type: ignore


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    error = pyqtSignal(str)
    file_saved_as = pyqtSignal(str)


class Generator(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handle worker thread setup, signals
    and wrap-up.

    :param data: The data to add to the PDF for generating.
    """

    def _init_(self, data):
        super()._init_()
        self.data = data
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            outfile = "result.pdf"

            template = PdfReader("template.pdf", decompress=False).pages[0]
            template_obj = pagexobj(template)

            canvas = Canvas(outfile)

            xobj_name = makerl(canvas, template_obj)
            canvas.doForm(xobj_name)

            ystart = 443

            # Prepared by
            canvas.drawString(170, ystart, self.data['name'])

            # Date: Todays date
            today = datetime.today()
            canvas.drawString(410, ystart, today.strftime('%F'))

            # Device/Program Type
            canvas.drawString(230, ystart-28, self.data['program_type'])

            # Product code
            canvas.drawString(175, ystart-(2*28), self.data['product_code'])

            # Customer
            canvas.drawString(315, ystart-(2*28), self.data['customer'])

            # Vendor
            canvas.drawString(145, ystart-(3*28), self.data['vendor'])

            ystart = 250

            # Program Language
            canvas.drawString(210, ystart, "Python")

            canvas.drawString(430, ystart, self.data['n_errors'])

            comments = self.data['comments'].replace('\n', ' ')
            if comments:
                lines = textwrap.wrap(comments, width=65) # 45
                first_line = lines[0]
                remainder = ' '.join(lines[1:])

                lines = textwrap.wrap(remainder, 75) # 55
                lines = lines[:4]  # max lines, not including the first.

                canvas.drawString(155, 223, first_line)
                for n, l in enumerate(lines, 1):
                    canvas.drawString(80, 223 - (n*28), l)

            canvas.save()

        except Exception as e:
            self.signals.error.emit(str(e))
            return

        self.signals.file_saved_as.emit(outfile)


class Window(QWidget):

    def _init_(self):
        super()._init_()

        self.threadpool = QThreadPool()

        self.name = QLineEdit()
        self.program_type = QLineEdit()
        self.product_code = QLineEdit()
        self.customer = QLineEdit()
        self.vendor = QLineEdit()
        self.n_errors = QSpinBox()
        self.n_errors.setRange(0, 1000)
        self.comments = QTextEdit()

        self.generate_btn = QPushButton("Generate PDF")
        self.generate_btn.pressed.connect(self.generate)

        layout = QFormLayout()
        layout.addRow("Name", self.name)
        layout.addRow("Program Type", self.program_type)
        layout.addRow("Product Code", self.product_code)
        layout.addRow("Customer", self.customer)
        layout.addRow("Vendor", self.vendor)
        layout.addRow("No. of Errors", self.n_errors)

        layout.addRow("Comments", self.comments)
        layout.addRow(self.generate_btn)

        self.setLayout(layout)

    def generate(self):
        self.generate_btn.setDisabled(True)
        data = {
            'name': self.name.text(),
            'program_type': self.program_type.text(),
            'product_code': self.product_code.text(),
            'customer': self.customer.text(),
            'vendor': self.vendor.text(),
            'n_errors': str(self.n_errors.value()),
            'comments': self.comments.toPlainText()
        }
        g = Generator(data)
        g.signals.file_saved_as.connect(self.generated)
        g.signals.error.connect(print)  # Print errors to console.
        self.threadpool.start(g)

    def generated(self, outfile):
        self.generate_btn.setDisabled(False)
        try:
            os.startfile(outfile)
        except Exception:
            # If startfile not available, show dialog.
            QMessageBox.information(self, "Finished", "PDF has been generated")


app = QApplication([])
w = Window()
w.show()
app.exec_()
