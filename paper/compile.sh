rm -rf compiled

mkdir compiled

pdflatex -output-directory="compiled" paper.tex
bibtex compiled/paper.aux
pdflatex -output-directory="compiled" paper.tex
pdflatex -output-directory="compiled" paper.tex


