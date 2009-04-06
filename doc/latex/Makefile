all: clean refman.dvi

ps: refman.ps

pdf: refman.pdf

ps_2on1: refman_2on1.ps

pdf_2on1: refman_2on1.pdf

refman.ps: refman.dvi
	dvips -o refman.ps refman.dvi

refman.pdf: refman.ps
	ps2pdf refman.ps refman.pdf

refman.dvi: refman.tex doxygen.sty
	echo "Running latex..."
	latex refman.tex
	echo "Running makeindex..."
	makeindex refman.idx
	echo "Rerunning latex...."
	latex refman.tex
	latex_count=5 ; \
	while egrep -s 'Rerun (LaTeX|to get cross-references right)' refman.log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      latex refman.tex ;\
	      latex_count=`expr $$latex_count - 1` ;\
	    done

refman_2on1.ps: refman.ps
	psnup -2 refman.ps >refman_2on1.ps

refman_2on1.pdf: refman_2on1.ps
	ps2pdf refman_2on1.ps refman_2on1.pdf

clean:
	rm -f *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out refman.pdf
