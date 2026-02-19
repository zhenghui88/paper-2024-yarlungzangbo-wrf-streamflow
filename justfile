main:
    pandoc --verbose --filter pandoc-crossref --citeproc --number-sections --csl style-agu.csl --reference-doc template-agu.docx main.md -o main.docx

si:
    pandoc --verbose --filter pandoc-crossref --citeproc --number-sections --csl style-agu.csl --reference-doc template-agu.docx si.md -o si.docx

all: main si
