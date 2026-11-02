"""Render every final-report page and check text bounds/content."""
from pathlib import Path
import json
import hashlib
import pypdfium2 as pdfium
import pdfplumber
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
def main(stem='gemma_final_assessment'):
    source=ROOT/('output/pdf/'+stem+'.pdf');out=ROOT/('tmp/pdfs/'+stem);out.mkdir(parents=True,exist_ok=True)
    doc=pdfium.PdfDocument(source);thumbs=[];pages=[]
    with pdfplumber.open(source) as pdf:
        for i,page in enumerate(pdf.pages):
            text=page.extract_text() or '';assert len(text)>250,(i,'almost empty page')
            errors=[c for c in page.chars if c['x0']<45 or c['x1']>550 or c['top']<25 or c['bottom']>815]
            assert not errors,(i,'text outside safe bounds',errors[:1])
            image=doc[i].render(scale=1.3).to_pil().convert('RGB');image.save(out/f'page-{i+1}.png')
            thumb=ImageOps.contain(image,(380,540));panel=Image.new('RGB',(400,570),'#dfe6eb');panel.paste(thumb,((400-thumb.width)//2,10));ImageDraw.Draw(panel).text((12,550),f'Page {i+1}',fill='black');thumbs.append(panel)
            pages.append({'page':i+1,'characters':len(text),'first_lines':text.splitlines()[:2]})
    contact=Image.new('RGB',(4*400,((len(thumbs)+3)//4)*570),'white')
    for i,t in enumerate(thumbs):contact.paste(t,((i%4)*400,(i//4)*570))
    contact.save(out/'contact.png')
    audit={'pdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pages':pages,'all_pages_rendered':True,'text_bounds_passed':True,'visual_review':'pending'}
    (ROOT/('output/pdf/'+stem+'_verification.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(pages,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stem',default='gemma_final_assessment');a=p.parse_args();main(a.stem)
