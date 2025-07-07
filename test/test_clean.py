def clean_svg(svg_text):
    if svg_text.lstrip().startswith("```svg"):
        svg_text = svg_text.lstrip()[6:]
    # Remove the end ```
    if svg_text.rstrip().endswith("```"):
        svg_text = svg_text.rstrip()[:-3]
    svg_text = svg_text.strip()

    return svg_text


if __name__ == "__main__":
    svg_text = """```svg
<svg width="100" height="100">
    <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
</svg>
abc```"""
    print(clean_svg(svg_text))