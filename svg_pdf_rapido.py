"""Exporta los elementos SVG usados por las gráficas de Aguas y Aguas a PDF vectorial.
Sin svglib: utiliza ReportLab, ya instalado en el proyecto.
"""
from __future__ import annotations
import re
from io import BytesIO
from xml.etree import ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor

_NUM = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
_PATH = re.compile(r'[MLZmlz]|' + _NUM)

def _num(value, default=0):
    try: return float(re.search(_NUM, str(value)).group())
    except (AttributeError, ValueError, TypeError): return default

def _color(value):
    if not value or value in ('none', 'transparent'): return None
    value = value.strip()
    if re.fullmatch(r'#[0-9a-fA-F]{3}', value):
        value = '#' + ''.join(ch * 2 for ch in value[1:])
    try: return HexColor(value)
    except (ValueError, TypeError): return None

def _path(c, d):
    tokens = _PATH.findall(d)
    p = c.beginPath(); cmd = None; i = 0; px = py = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]; i += 1
            if cmd.lower() == 'z': p.close(); continue
        if cmd is None or i + 1 >= len(tokens): break
        x, y = float(tokens[i]), float(tokens[i + 1]); i += 2
        if cmd.islower(): x += px; y += py
        if cmd.lower() == 'm': p.moveTo(x, y); cmd = 'l' if cmd == 'm' else 'L'
        elif cmd.lower() == 'l': p.lineTo(x, y)
        px, py = x, y
    return p

def _render(c, el, inherited):
    kind = el.tag.split('}')[-1]
    if kind in ('defs', 'title', 'desc'): return
    attrs = {**inherited, **el.attrib}
    if kind in ('svg', 'g'):
        for child in el: _render(c, child, attrs)
        return
    if kind not in ('rect', 'line', 'path', 'circle', 'text'): return
    c.saveState()
    fill = _color(attrs.get('fill', '#000000'))
    stroke = _color(attrs.get('stroke'))
    c.setFillColor(fill or HexColor('#000000'))
    c.setStrokeColor(stroke or HexColor('#000000'))
    c.setLineWidth(_num(attrs.get('stroke-width'), 1))
    opacity = _num(attrs.get('opacity'), 1)
    if hasattr(c, 'setFillAlpha'): c.setFillAlpha(opacity * _num(attrs.get('fill-opacity'), 1))
    if hasattr(c, 'setStrokeAlpha'): c.setStrokeAlpha(opacity * _num(attrs.get('stroke-opacity'), 1))
    dash = attrs.get('stroke-dasharray', '')
    if dash not in ('', 'none'):
        numbers = [_num(x) for x in re.split(r'[,\s]+', dash.strip()) if x]
        if numbers: c.setDash(numbers)
    if kind == 'rect':
        x, y = _num(attrs.get('x')), _num(attrs.get('y'))
        w, h = _num(attrs.get('width')), _num(attrs.get('height'))
        if w > 0 and h > 0: c.rect(x, y, w, h, fill=int(fill is not None), stroke=int(stroke is not None))
    elif kind == 'line':
        if stroke is not None: c.line(*[_num(attrs.get(k)) for k in ('x1','y1','x2','y2')])
    elif kind == 'circle':
        c.circle(_num(attrs.get('cx')), _num(attrs.get('cy')), _num(attrs.get('r')), fill=int(fill is not None), stroke=int(stroke is not None))
    elif kind == 'path':
        c.drawPath(_path(c, attrs.get('d', '')), fill=int(fill is not None), stroke=int(stroke is not None))
    elif kind == 'text':
        x, y = _num(attrs.get('x')), _num(attrs.get('y'))
        size = _num(attrs.get('font-size'), 12)
        weight = attrs.get('font-weight', '')
        font = 'Helvetica-Bold' if weight in ('600', '700', 'bold', '800') else 'Helvetica'
        c.setFont(font, size)
        transform = attrs.get('transform', '')
        match = re.match(r'rotate\(\s*(' + _NUM + r')(?:[,\s]+(' + _NUM + r')[,\s]+(' + _NUM + r'))?\s*\)', transform)
        if match:
            angle = float(match.group(1)); cx = float(match.group(2) or 0); cy = float(match.group(3) or 0)
            c.translate(cx, cy); c.rotate(angle); x -= cx; y -= cy
        # El SVG usa eje Y hacia abajo; la página ya fue invertida para
        # figuras, pero el texto necesita una segunda inversión local.
        c.translate(x, y)
        c.scale(1, -1)
        s = ''.join(el.itertext())
        align = attrs.get('text-anchor')
        if align == 'middle': c.drawCentredString(0, 0, s)
        elif align == 'end': c.drawRightString(0, 0, s)
        else: c.drawString(0, 0, s)
    c.restoreState()

def generar_pdf_vectorial(svg_texto, titulo, subtitulo):
    raiz = ET.fromstring(svg_texto)
    W, H = _num(raiz.get('width')), _num(raiz.get('height'))
    if W <= 0 or H <= 0: raise ValueError('Dimensiones de SVG inválidas')
    out = BytesIO()
    pw, ph = landscape(A4); margen = 30; titulo_h = 65
    pdf = canvas.Canvas(out, pagesize=(pw, ph))
    pdf.setFont('Helvetica-Bold', 13); pdf.setFillColor(HexColor('#0b3d7a'))
    pdf.drawString(margen, ph - 32, titulo)
    pdf.setFont('Helvetica', 8); pdf.setFillColor(HexColor('#727272'))
    while subtitulo and pdf.stringWidth(subtitulo, 'Helvetica', 8) > pw - 2*margen: subtitulo = subtitulo[:-1]
    pdf.drawString(margen, ph - 49, subtitulo)
    escala = min((pw - 2*margen)/W, (ph - titulo_h - margen)/H)
    x0 = (pw-W*escala)/2; y0 = ph-titulo_h-H*escala
    pdf.saveState(); pdf.translate(x0, y0+H*escala); pdf.scale(escala, -escala)
    for child in raiz: _render(pdf, child, raiz.attrib)
    pdf.restoreState(); pdf.save()
    return out.getvalue()
