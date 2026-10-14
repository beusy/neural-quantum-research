# Author: Don van Beusichem
"""Build standalone LaTeX and PDF; keep each short table with its caption."""
from pathlib import Path
import subprocess,re
p=Path(__file__).parent
subprocess.run(['pandoc','paper.md','-s','-t','latex','--toc','-V','documentclass=article','-V','papersize=a4','-o','main.tex'],cwd=p,check=True)
s=(p/'main.tex').read_text()
pattern=r'\\needspace\{12\\baselineskip\}(.*?)\\begin\{longtable\}\[\]\{(.*?)\\end\{longtable\}'
def table(m):
 caption=m[1];body=m[2]
 body=body.replace('\\endhead','').replace('\\bottomrule\\noalign{}\n\\endlastfoot','')
 return '\\par\\medskip\n\\noindent\\begin{minipage}{\\linewidth}'+caption+'\\begin{tabular}{'+body+'\\bottomrule\n\\end{tabular}\n\\end{minipage}\\par\\medskip\n'
s,n=re.subn(pattern,table,s,flags=re.S);assert n==9,n
(p/'main.tex').write_text(s)
for _ in range(3):
 r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=p,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (p/'pdf_build.log').write_bytes(r.stdout)
 if r.returncode: raise RuntimeError(r.stdout.decode()[-3000:])
(p/'diagnosing_neural_quantum_states.pdf').write_bytes((p/'main.pdf').read_bytes())
