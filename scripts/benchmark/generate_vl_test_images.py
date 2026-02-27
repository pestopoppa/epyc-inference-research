#!/usr/bin/env python3
from __future__ import annotations

"""Generate test images for VL quality rubric."""

import os

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "/mnt/raid0/llm/epyc-inference-research/test_images/vl_rubric"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_font(size=40):
    """Get a font, falling back to default if needed."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception as e:
        return ImageFont.load_default()

# T1-Q1: Simple OCR
def create_text_simple():
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(48)
    draw.text((20, 25), "Hello World 123", fill='black', font=font)
    img.save(f"{OUTPUT_DIR}/text_simple.png")
    print("Created: text_simple.png")

# T1-Q2: Basic shapes
def create_shapes_basic():
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    # 3 red circles
    for i, x in enumerate([50, 120, 190]):
        draw.ellipse([x, 50, x+40, 90], fill='red', outline='darkred')
    # 2 blue squares
    for i, x in enumerate([280, 340]):
        draw.rectangle([x, 50, x+40, 90], fill='blue', outline='darkblue')
    img.save(f"{OUTPUT_DIR}/shapes_basic.png")
    print("Created: shapes_basic.png")

# T1-Q3: Folder icon (simple representation)
def create_icon_folder():
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    # Folder shape
    draw.rectangle([10, 30, 90, 85], fill='#FFD700', outline='#DAA520')
    draw.rectangle([10, 20, 40, 35], fill='#FFD700', outline='#DAA520')
    img.save(f"{OUTPUT_DIR}/icon_folder.png")
    print("Created: icon_folder.png")

# T2-Q1: Bar chart
def create_chart_bar():
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(20)

    # Axis
    draw.line([(50, 250), (350, 250)], fill='black', width=2)
    draw.line([(50, 250), (50, 30)], fill='black', width=2)

    # Bars: A=10, B=25, C=15, D=20
    values = [('A', 10), ('B', 25), ('C', 15), ('D', 20)]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    for i, (label, val) in enumerate(values):
        x = 80 + i * 70
        height = val * 8
        draw.rectangle([x, 250-height, x+50, 250], fill=colors[i])
        draw.text((x+15, 255), label, fill='black', font=font)
        draw.text((x+10, 250-height-25), str(val), fill='black', font=font)

    # Y-axis labels
    for v in [0, 10, 20, 30]:
        y = 250 - v * 8
        draw.text((25, y-10), str(v), fill='black', font=font)

    img.save(f"{OUTPUT_DIR}/chart_bar.png")
    print("Created: chart_bar.png")

# T2-Q2: Simple invoice
def create_doc_invoice():
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    font_bold = get_font(20)

    draw.text((150, 10), "INVOICE", fill='black', font=font_bold)
    draw.text((20, 50), "Date: 2025-12-16", fill='black', font=font)
    draw.line([(20, 80), (380, 80)], fill='black')

    # Header
    draw.text((20, 90), "Item", fill='black', font=font)
    draw.text((150, 90), "Qty", fill='black', font=font)
    draw.text((220, 90), "Price", fill='black', font=font)
    draw.text((300, 90), "Amount", fill='black', font=font)
    draw.line([(20, 115), (380, 115)], fill='black')

    # Items
    items = [("Widget A", 2, 15.00), ("Widget B", 3, 25.00), ("Service", 1, 50.00)]
    y = 125
    for item, qty, price in items:
        amount = qty * price
        draw.text((20, y), item, fill='black', font=font)
        draw.text((150, y), str(qty), fill='black', font=font)
        draw.text((220, y), f"${price:.2f}", fill='black', font=font)
        draw.text((300, y), f"${amount:.2f}", fill='black', font=font)
        y += 30

    draw.line([(20, y), (380, y)], fill='black')
    total = sum(qty * price for _, qty, price in items)
    draw.text((220, y+10), "TOTAL:", fill='black', font=font_bold)
    draw.text((300, y+10), f"${total:.2f}", fill='black', font=font_bold)

    img.save(f"{OUTPUT_DIR}/doc_invoice.png")
    print("Created: doc_invoice.png")

# T2-Q3: Code with bug
def create_code_python():
    img = Image.new('RGB', (500, 250), color='#1e1e1e')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except Exception as e:
        font = get_font(16)

    code_lines = [
        ("def ", "#569cd6"), ("calculate_average", "#dcdcaa"), ("(numbers):", "#d4d4d4"),
        ("    total = ", "#d4d4d4"), ("0", "#b5cea8"),
        ("    for ", "#c586c0"), ("num ", "#9cdcfe"), ("in ", "#c586c0"), ("numbers:", "#d4d4d4"),
        ("        total += num", "#d4d4d4"),
        ("    average = total / ", "#d4d4d4"), ("len(numbers)", "#dcdcaa"),
        ("    return ", "#c586c0"), ("total", "#9cdcfe"), ("  # BUG: should return average", "#6a9955"),
    ]

    y = 20
    for i, line in enumerate(code_lines):
        if isinstance(line, tuple) and len(line) == 2:
            draw.text((20, y), line[0], fill=line[1], font=font)
        else:
            x = 20
            for text, color in [line[j:j+2] for j in range(0, len(line), 2)]:
                draw.text((x, y), text, fill=color, font=font)
                x += len(text) * 10
        y += 25

    img.save(f"{OUTPUT_DIR}/code_python.png")
    print("Created: code_python.png")

# T3-Q1: Math equation
def create_math_equation():
    img = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(36)
    draw.text((30, 30), "2x + 5 = 13", fill='black', font=font)
    # Add "solve for x" instruction
    font_small = get_font(20)
    draw.text((30, 75), "Solve for x", fill='gray', font=font_small)
    img.save(f"{OUTPUT_DIR}/math_equation.png")
    print("Created: math_equation.png")

# T3-Q2: Flowchart
def create_diagram_flowchart():
    img = Image.new('RGB', (500, 400), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(14)

    # Start
    draw.ellipse([200, 10, 300, 50], outline='black', width=2)
    draw.text((225, 20), "START", fill='black', font=font)

    # Diamond: input > 10?
    draw.polygon([(250, 80), (320, 130), (250, 180), (180, 130)], outline='black', width=2)
    draw.text((210, 115), "input > 10?", fill='black', font=font)

    # Yes branch - Diamond: flag = true?
    draw.polygon([(380, 130), (450, 180), (380, 230), (310, 180)], outline='black', width=2)
    draw.text((335, 165), "flag=true?", fill='black', font=font)

    # No branch from first diamond
    draw.rectangle([50, 110, 150, 150], outline='black', width=2)
    draw.text((65, 120), "Path A", fill='black', font=font)

    # Yes from second diamond
    draw.rectangle([350, 270, 450, 310], outline='black', width=2)
    draw.text((365, 280), "Path B", fill='black', font=font)

    # No from second diamond
    draw.rectangle([200, 270, 300, 310], outline='black', width=2)
    draw.text((215, 280), "Path C", fill='black', font=font)

    # End
    draw.ellipse([200, 340, 300, 380], outline='black', width=2)
    draw.text((230, 350), "END", fill='black', font=font)

    # Arrows
    draw.line([(250, 50), (250, 80)], fill='black', width=2)
    draw.line([(320, 130), (380, 130)], fill='black', width=2)  # Yes to second diamond
    draw.text((335, 110), "Yes", fill='green', font=font)
    draw.line([(180, 130), (150, 130)], fill='black', width=2)  # No to Path A
    draw.text((155, 110), "No", fill='red', font=font)
    draw.line([(380, 230), (380, 270)], fill='black', width=2)  # Yes to Path B
    draw.text((385, 240), "Yes", fill='green', font=font)
    draw.line([(310, 180), (250, 180), (250, 270)], fill='black', width=2)  # No to Path C
    draw.text((260, 220), "No", fill='red', font=font)
    draw.line([(250, 310), (250, 340)], fill='black', width=2)
    draw.line([(400, 310), (400, 360), (300, 360)], fill='black', width=2)
    draw.line([(100, 150), (100, 360), (200, 360)], fill='black', width=2)

    img.save(f"{OUTPUT_DIR}/diagram_flowchart.png")
    print("Created: diagram_flowchart.png")

# T3-Q3: Spot the difference
def create_diff_images():
    # Create two images side by side with 3 differences
    img = Image.new('RGB', (500, 200), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    # Left image
    draw.rectangle([20, 20, 220, 180], outline='black')
    draw.text((90, 5), "Image A", fill='black', font=font)
    draw.ellipse([50, 50, 100, 100], fill='red')  # Red circle
    draw.rectangle([120, 50, 170, 100], fill='blue')  # Blue square
    draw.ellipse([80, 120, 130, 160], fill='green')  # Green circle

    # Right image (3 differences)
    draw.rectangle([280, 20, 480, 180], outline='black')
    draw.text((350, 5), "Image B", fill='black', font=font)
    draw.ellipse([310, 50, 360, 100], fill='yellow')  # Diff 1: yellow instead of red
    draw.rectangle([380, 50, 430, 100], fill='blue')  # Same blue square
    draw.rectangle([340, 120, 390, 160], fill='green')  # Diff 2: square instead of circle
    # Diff 3: missing element (no third shape in same position)
    draw.ellipse([400, 130, 440, 170], fill='purple')  # Extra purple circle

    img.save(f"{OUTPUT_DIR}/diff_images.png")
    print("Created: diff_images.png")

# T3-Q4: Pattern puzzle
def create_puzzle_grid():
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(20)

    # 3x3 grid
    cell_size = 80
    offset = 30

    for i in range(4):
        draw.line([(offset, offset + i*cell_size), (offset + 3*cell_size, offset + i*cell_size)], fill='black', width=2)
        draw.line([(offset + i*cell_size, offset), (offset + i*cell_size, offset + 3*cell_size)], fill='black', width=2)

    # Pattern: Row 1: circle, square, triangle
    #          Row 2: square, triangle, circle
    #          Row 3: triangle, circle, ?
    shapes = [
        ['circle', 'square', 'triangle'],
        ['square', 'triangle', 'circle'],
        ['triangle', 'circle', '?']
    ]

    for row in range(3):
        for col in range(3):
            x = offset + col * cell_size + cell_size // 2
            y = offset + row * cell_size + cell_size // 2
            shape = shapes[row][col]

            if shape == 'circle':
                draw.ellipse([x-25, y-25, x+25, y+25], outline='blue', width=3)
            elif shape == 'square':
                draw.rectangle([x-25, y-25, x+25, y+25], outline='red', width=3)
            elif shape == 'triangle':
                draw.polygon([(x, y-25), (x-25, y+25), (x+25, y+25)], outline='green', width=3)
            elif shape == '?':
                draw.text((x-10, y-15), "?", fill='gray', font=get_font(40))

    img.save(f"{OUTPUT_DIR}/puzzle_grid.png")
    print("Created: puzzle_grid.png")

# T3 Expert: Scientific figure with multiple panels and error bars
def create_scientific_figure():
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    font_title = get_font(18)
    font_small = get_font(12)

    # Title
    draw.text((300, 10), "Figure 3: Treatment Effects", fill='black', font=font_title)

    # Panel A - Bar chart with error bars (top left)
    draw.text((50, 40), "A) Gene Expression Levels", fill='black', font=font)
    draw.rectangle([50, 60, 350, 280], outline='black')
    # Y-axis
    draw.line([(80, 260), (80, 80)], fill='black', width=2)
    draw.line([(80, 260), (320, 260)], fill='black', width=2)
    for i, label in enumerate(['0', '50', '100', '150']):
        y = 260 - i * 45
        draw.text((55, y-8), label, fill='black', font=font_small)
        draw.line([(75, y), (80, y)], fill='black')

    # Bars with error bars
    conditions = [('Control', 45, 8, '#888888'), ('Drug A', 95, 15, '#4CAF50'),
                  ('Drug B', 120, 12, '#2196F3'), ('Combo', 85, 25, '#FF9800')]
    for i, (label, val, err, color) in enumerate(conditions):
        x = 100 + i * 55
        bar_height = val * 1.2
        draw.rectangle([x, 260-bar_height, x+40, 260], fill=color)
        # Error bar
        draw.line([(x+20, 260-bar_height-err), (x+20, 260-bar_height+err)], fill='black', width=2)
        draw.line([(x+15, 260-bar_height-err), (x+25, 260-bar_height-err)], fill='black', width=2)
        draw.line([(x+15, 260-bar_height+err), (x+25, 260-bar_height+err)], fill='black', width=2)
        draw.text((x, 265), label, fill='black', font=font_small)

    # Note: asterisks for significance
    draw.text((155, 260-95*1.2-25), "*", fill='black', font=get_font(24))
    draw.text((210, 260-120*1.2-22), "**", fill='black', font=get_font(24))

    # Panel B - Line chart (top right)
    draw.text((420, 40), "B) Time Course Analysis", fill='black', font=font)
    draw.rectangle([420, 60, 750, 280], outline='black')
    draw.line([(450, 260), (450, 80)], fill='black', width=2)
    draw.line([(450, 260), (720, 260)], fill='black', width=2)

    # Time points with two lines
    points_a = [(470, 240), (520, 200), (570, 150), (620, 130), (670, 125)]
    points_b = [(470, 235), (520, 220), (570, 210), (620, 200), (670, 190)]
    for i in range(len(points_a)-1):
        draw.line([points_a[i], points_a[i+1]], fill='#2196F3', width=2)
        draw.line([points_b[i], points_b[i+1]], fill='#FF5722', width=2)
    for p in points_a:
        draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill='#2196F3')
    for p in points_b:
        draw.rectangle([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill='#FF5722')

    # Legend
    draw.rectangle([680, 90, 740, 130], outline='gray')
    draw.line([(685, 100), (705, 100)], fill='#2196F3', width=2)
    draw.text((710, 93), "Tx", fill='black', font=font_small)
    draw.line([(685, 115), (705, 115)], fill='#FF5722', width=2)
    draw.text((710, 108), "Ctrl", fill='black', font=font_small)

    # Panel C - Scatter plot (bottom left)
    draw.text((50, 310), "C) Correlation Analysis", fill='black', font=font)
    draw.rectangle([50, 330, 350, 550], outline='black')
    draw.line([(80, 530), (80, 350)], fill='black', width=2)
    draw.line([(80, 530), (320, 530)], fill='black', width=2)
    draw.text((60, 335), "Y", fill='black', font=font_small)
    draw.text((310, 535), "X", fill='black', font=font_small)

    # Scatter points with trend line
    import random
    random.seed(42)
    for _ in range(25):
        x = random.randint(100, 300)
        y = 530 - int((x - 80) * 0.6) + random.randint(-30, 30)
        draw.ellipse([x-3, y-3, x+3, y+3], fill='#9C27B0')
    # Trend line
    draw.line([(90, 520), (310, 380)], fill='red', width=2)
    draw.text((200, 360), "r = 0.78, p < 0.001", fill='black', font=font_small)

    # Panel D - Heatmap (bottom right) - CONTROVERSIAL: no scale bar
    draw.text((420, 310), "D) Expression Heatmap", fill='black', font=font)
    draw.rectangle([420, 330, 750, 550], outline='black')
    colors_heat = ['#0000FF', '#4444FF', '#8888FF', '#FFFFFF', '#FF8888', '#FF4444', '#FF0000']
    for row in range(6):
        for col in range(8):
            c = colors_heat[random.randint(0, 6)]
            x = 440 + col * 35
            y = 350 + row * 30
            draw.rectangle([x, y, x+30, y+25], fill=c, outline='gray')
    # Gene labels (missing for some - methodological issue)
    genes = ['Gene1', 'Gene2', '', 'Gene4', '', 'Gene6']
    for i, g in enumerate(genes):
        draw.text((750, 355 + i * 30), g, fill='black', font=font_small)

    # Missing: no color scale legend (methodological issue to identify)

    img.save(f"{OUTPUT_DIR}/scientific_figure.png")
    print("Created: scientific_figure.png")


# T3 Expert: Architectural blueprint
def create_blueprint():
    img = Image.new('RGB', (800, 600), color='#E8F4F8')  # Light blueprint blue
    draw = ImageDraw.Draw(img)
    font = get_font(12)
    font_small = get_font(10)

    # Title block
    draw.rectangle([600, 520, 790, 590], outline='black', width=2)
    draw.text((620, 530), "Floor Plan - Level 1", fill='black', font=font)
    draw.text((620, 550), "Scale: 1/4\" = 1'", fill='black', font=font_small)
    draw.text((620, 565), "Date: 2025-01-01", fill='black', font=font_small)

    # Main structure outline
    draw.rectangle([50, 50, 550, 450], outline='black', width=3)

    # Rooms
    # Living room (large)
    draw.rectangle([50, 50, 300, 250], outline='black', width=2)
    draw.text((140, 140), "LIVING ROOM", fill='black', font=font)
    draw.text((140, 160), "18' x 20'", fill='gray', font=font_small)

    # Kitchen
    draw.rectangle([300, 50, 450, 200], outline='black', width=2)
    draw.text((340, 110), "KITCHEN", fill='black', font=font)
    draw.text((340, 130), "12' x 15'", fill='gray', font=font_small)
    # Counter (L-shaped)
    draw.rectangle([305, 55, 320, 150], fill='#A0A0A0')
    draw.rectangle([305, 55, 400, 70], fill='#A0A0A0')
    # Stove symbol
    draw.rectangle([350, 55, 380, 70], fill='#FFD700', outline='black')
    draw.text((355, 56), "S", fill='black', font=font_small)

    # Bedroom 1
    draw.rectangle([50, 250, 200, 450], outline='black', width=2)
    draw.text((90, 340), "BEDROOM 1", fill='black', font=font)
    draw.text((90, 360), "12' x 20'", fill='gray', font=font_small)

    # Bedroom 2
    draw.rectangle([200, 250, 350, 450], outline='black', width=2)
    draw.text((235, 340), "BEDROOM 2", fill='black', font=font)
    draw.text((235, 360), "12' x 20'", fill='gray', font=font_small)

    # Bathroom (CODE VIOLATION: no window, single egress)
    draw.rectangle([350, 300, 450, 450], outline='black', width=2)
    draw.text((370, 360), "BATH", fill='black', font=font)
    draw.text((370, 380), "8' x 15'", fill='gray', font=font_small)
    # Toilet
    draw.ellipse([400, 400, 430, 430], outline='black')
    # Sink
    draw.rectangle([360, 310, 390, 340], outline='black')

    # Hallway
    draw.rectangle([450, 200, 550, 450], outline='black', width=2)
    draw.text((470, 320), "HALL", fill='black', font=font)

    # Utility room (unlabeled - potential issue)
    draw.rectangle([450, 50, 550, 200], outline='black', width=2)

    # Doors (door swings shown)
    # Front door
    draw.arc([240, 430, 280, 470], 0, 90, fill='black', width=2)
    draw.line([(260, 450), (260, 470)], fill='black', width=3)

    # Interior doors
    draw.arc([195, 230, 225, 260], 270, 360, fill='black', width=2)
    draw.arc([345, 280, 375, 310], 270, 360, fill='black', width=2)

    # Windows (double lines)
    for x in [100, 400]:
        draw.line([(x, 50), (x+40, 50)], fill='blue', width=4)
    for y in [100, 350]:
        draw.line([(50, y), (50, y+40)], fill='blue', width=4)

    # Electrical symbols (RED overlay)
    # Outlets
    outlets = [(80, 100), (180, 100), (320, 80), (100, 280), (230, 280), (380, 320)]
    for x, y in outlets:
        draw.ellipse([x-5, y-5, x+5, y+5], outline='red', width=2)
    # Light fixtures
    lights = [(170, 150), (370, 130), (130, 350), (270, 350), (390, 370)]
    for x, y in lights:
        draw.ellipse([x-8, y-8, x+8, y+8], outline='red', fill='yellow')

    # HVAC duct (GREEN overlay) - ISSUE: no return in bedroom 2
    draw.rectangle([500, 100, 530, 180], outline='green', width=2)
    draw.text((505, 140), "AC", fill='green', font=font_small)
    # Supply vents
    for x, y in [(170, 200), (370, 180), (100, 400)]:
        draw.rectangle([x-10, y-5, x+10, y+5], fill='green')

    # Plumbing (BLUE symbols) - water heater in utility
    draw.ellipse([480, 80, 520, 120], outline='blue', fill='#ADD8E6')
    draw.text((485, 90), "WH", fill='blue', font=font_small)

    # North arrow
    draw.polygon([(750, 100), (730, 150), (750, 140), (770, 150)], fill='black')
    draw.text((742, 155), "N", fill='black', font=font)

    img.save(f"{OUTPUT_DIR}/blueprint.png")
    print("Created: blueprint.png")


# T3 Expert: Satellite sequence showing temporal changes
def create_satellite_sequence():
    import random
    random.seed(123)

    img = Image.new('RGB', (900, 400), color='white')
    draw = ImageDraw.Draw(img)
    font = get_font(12)
    font_title = get_font(14)

    draw.text((350, 10), "Satellite Imagery: 2015-2025", fill='black', font=font_title)

    years = ['2015', '2017', '2019', '2021', '2023', '2025']

    for idx, year in enumerate(years):
        x_offset = 10 + idx * 148
        # Frame
        draw.rectangle([x_offset, 40, x_offset + 140, 180], outline='black', width=2)
        draw.text((x_offset + 55, 185), year, fill='black', font=font)

        # Base terrain (green for vegetation, brown for bare)
        base_green = max(0, 180 - idx * 25)  # Decreasing vegetation
        for y in range(45, 175, 10):
            for x in range(x_offset + 5, x_offset + 135, 10):
                # Random terrain with trend
                if random.random() < (0.8 - idx * 0.1):
                    color = (50, base_green + random.randint(-20, 20), 50)
                else:
                    color = (139, 119, 101)  # Brown
                draw.rectangle([x, y, x+8, y+8], fill=color)

        # Water body (shrinking over time)
        water_size = max(5, 40 - idx * 6)
        draw.ellipse([x_offset + 80 - water_size//2, 100 - water_size//2,
                      x_offset + 80 + water_size//2, 100 + water_size//2], fill='#4169E1')

        # Urban expansion (growing over time)
        urban_count = idx * 3
        for _ in range(urban_count):
            ux = x_offset + 20 + random.randint(0, 80)
            uy = 60 + random.randint(0, 90)
            draw.rectangle([ux, uy, ux + 8, uy + 8], fill='#808080')

        # Road (appears in 2019)
        if idx >= 2:
            draw.line([(x_offset + 20, 120), (x_offset + 120, 90)], fill='#404040', width=3)

        # Industrial area (appears in 2021)
        if idx >= 3:
            draw.rectangle([x_offset + 90, 140, x_offset + 130, 170], fill='#696969', outline='black')

    # Legend
    draw.rectangle([10, 220, 200, 380], outline='black')
    draw.text((20, 225), "Legend:", fill='black', font=font)
    draw.rectangle([20, 250, 40, 270], fill='#228B22')
    draw.text((50, 252), "Vegetation", fill='black', font=font)
    draw.rectangle([20, 280, 40, 300], fill='#8B7355')
    draw.text((50, 282), "Bare soil", fill='black', font=font)
    draw.ellipse([20, 310, 40, 330], fill='#4169E1')
    draw.text((50, 312), "Water body", fill='black', font=font)
    draw.rectangle([20, 340, 40, 360], fill='#808080')
    draw.text((50, 342), "Urban/developed", fill='black', font=font)

    # Analysis box
    draw.rectangle([220, 220, 890, 380], outline='black')
    draw.text((230, 225), "Key Observations:", fill='black', font=font)
    observations = [
        "1. Vegetation cover decreased ~60% (2015-2025)",
        "2. Water body shrunk by approximately 70%",
        "3. Urban area expanded significantly after 2019",
        "4. New road infrastructure visible from 2019",
        "5. Industrial development began ~2021",
        "Environmental concerns: Deforestation, water depletion"
    ]
    for i, obs in enumerate(observations):
        draw.text((240, 250 + i * 20), obs, fill='black', font=font)

    img.save(f"{OUTPUT_DIR}/satellite_sequence.png")
    print("Created: satellite_sequence.png")


if __name__ == "__main__":
    print(f"Generating test images in {OUTPUT_DIR}/\n")

    # T1
    create_text_simple()
    create_shapes_basic()
    create_icon_folder()

    # T2
    create_chart_bar()
    create_doc_invoice()
    create_code_python()

    # T3
    create_math_equation()
    create_diagram_flowchart()
    create_diff_images()
    create_puzzle_grid()

    # T3 Expert
    create_scientific_figure()
    create_blueprint()
    create_satellite_sequence()

    print(f"\nDone! {len(os.listdir(OUTPUT_DIR))} images created.")
