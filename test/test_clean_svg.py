from starvector.data.util import clean_svg, use_placeholder
from svgpathtools import svgstr2paths
from cairosvg import svg2svg
from bs4 import BeautifulSoup


def post_process_svg(text):
    """Post-process a single SVG text"""
    try:
        svgstr2paths(text)
        return {
            'svg': text,
            'svg_raw': text,
            'post_processed': False,
            # 'no_compile': False
            'non_compiling': False
        }
    except:
        try:
            cleaned_svg = clean_svg(text)
            print(cleaned_svg)
            svgstr2paths(cleaned_svg)
            return {
                'svg': cleaned_svg,
                'svg_raw': text,
                'post_processed': True,
                # 'no_compile': False
                'non_compiling': False
            }
        except:
            return {
                'svg': use_placeholder(),
                'svg_raw': text,
                'post_processed': True,
                # 'no_compile': True
                'non_compiling': True
            }


test_svg_code = """```svg
<svg width="100%" height="100%">
    <rect x="0" y="0" width="50%" height="100%" fill="green"/>
    <rect x="50%" y="0" width="50%" height="100%" fill="red"/>
    <circle cx="25%" cy="25%" r="20%" fill="yellow"/>
    <rect x="0" y="0" width="100%" height="50%" fill="white"/>
</svg>
```"""
cleaned_svg_code = """<svg width="100%" height="100%">
    <rect x="0" y="0" width="50%" height="100%" fill="green"/>
    <rect x="50%" y="0" width="50%" height="100%" fill="red"/>
    <circle cx="25%" cy="25%" r="20%" fill="yellow"/>
    <rect x="0" y="0" width="100%" height="50%" fill="white"/>
</svg>"""

soup = BeautifulSoup(cleaned_svg_code, 'xml') # Read as soup to parse as xml
svg_bs4 = soup.prettify()
svg_cairo = svg2svg(svg_bs4, output_width=None, output_height=None).decode()

# print(post_process_svg(test_svg_code))