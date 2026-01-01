import csv
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# --- 1. Data Preparation ---
csv_data = """Strategy,Paradigm,Workers,Time (s),Speedup,Efficiency
Pixel-Level,Multiprocessing,1,326.8898,0.9980,0.9980
Pixel-Level,Multiprocessing,2,170.5357,1.9131,0.9565
Pixel-Level,Multiprocessing,4,86.0291,3.7923,0.9481
Pixel-Level,Multiprocessing,8,48.9070,6.6708,0.8339
Pixel-Level,Multiprocessing,16,41.9884,7.7700,0.4856
Pixel-Level,Concurrent.Futures,1,323.8806,1.0073,1.0073
Pixel-Level,Concurrent.Futures,2,171.0455,1.9074,0.9537
Pixel-Level,Concurrent.Futures,4,82.9550,3.9329,0.9832
Pixel-Level,Concurrent.Futures,8,50.6574,6.4403,0.8050
Pixel-Level,Concurrent.Futures,16,45.4273,7.1818,0.4489
Image-Level,Multiprocessing,1,323.9240,1.0072,1.0072
Image-Level,Multiprocessing,2,181.5352,1.7972,0.8986
Image-Level,Multiprocessing,4,89.1194,3.6608,0.9152
Image-Level,Multiprocessing,8,52.5063,6.2136,0.7767
Image-Level,Multiprocessing,16,44.6422,7.3081,0.4568
Image-Level,Concurrent.Futures,1,328.8533,0.9921,0.9921
Image-Level,Concurrent.Futures,2,175.4021,1.8600,0.9300
Image-Level,Concurrent.Futures,4,90.2210,3.6161,0.9040
Image-Level,Concurrent.Futures,8,51.8890,6.2875,0.7859
Image-Level,Concurrent.Futures,16,46.1200,7.0739,0.4421"""

def parse_data(csv_text):
    data = {}
    reader = csv.reader(csv_text.strip().splitlines())
    next(reader) # skip header
    for row in reader:
        strategy, paradigm, workers, time, speedup, eff = row
        key = f"{strategy} ({paradigm})"
        if key not in data:
            data[key] = {'workers': [], 'speedup': [], 'efficiency': []}
        data[key]['workers'].append(int(workers))
        data[key]['speedup'].append(float(speedup))
        data[key]['efficiency'].append(float(eff))
    return data

# --- 2. Generate Chart ---
def create_chart(data, filename="chart.png"):
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (label, values) in enumerate(data.items()):
        plt.plot(values['workers'], values['speedup'], 
                 marker=markers[i%4], linestyle=linestyles[i%4], label=label, linewidth=2)
    
    # Ideal Speedup Line
    plt.plot([1, 16], [1, 16], 'k--', alpha=0.3, label="Ideal Linear Speedup")
    
    plt.title("Speedup vs Worker Count", fontsize=14)
    plt.xlabel("Number of Workers (CPUs)", fontsize=12)
    plt.ylabel("Speedup (x)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# --- 3. PDF Creation ---
def create_pdf(filename):
    # Prepare Data
    parsed_data = parse_data(csv_data)
    create_chart(parsed_data, "chart_generated.png")
    
    doc = SimpleDocTemplate(filename, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    # Styles
    title_style = styles['Title']
    h1 = styles['Heading1']
    h2 = styles['Heading2']
    h3 = styles['Heading3']
    normal = styles['Normal']
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Parallel Image Processing Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("CST435: Parallel and Cloud Computing", h2))
    story.append(Paragraph("Assignment 2: Food-101 Dataset Filtering", h3))
    story.append(PageBreak())

    # 1. Implementation
    story.append(Paragraph("1. Implementation", h1))
    story.append(Paragraph("1.1 Code Structure", h2))
    story.append(Paragraph("The solution is modularized into two primary Python files to separate orchestration from logic:", normal))
    
    bullets = [
        "<b>main.py (Orchestrator):</b> Manages the benchmarking loop. It iterates through worker counts (1, 2, 4, 8, 16), selects the execution strategy, and logs performance metrics using a custom <code>DualLogger</code>.",
        "<b>filters.py (Processing Logic):</b> Implements the five image filters (Grayscale, Blur, Sharpen, Edge Detection, Brightness). It contains specific worker functions (e.g., <code>_process_grayscale_chunk</code>) that handle shared memory arrays for the Pixel-Level strategy."
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", normal))
        story.append(Spacer(1, 6))

    story.append(Paragraph("1.2 Strategies & Paradigms", h2))
    story.append(Paragraph("<b>Strategies:</b>", normal))
    story.append(Paragraph("• <b>Pixel-Level:</b> Decomposes a single image into horizontal chunks processed in parallel. Optimized for high-resolution images to minimize single-image latency.", normal))
    story.append(Paragraph("• <b>Image-Level:</b> Distributes whole images to different workers. Optimized for high throughput of batch processing.", normal))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Paradigms:</b>", normal))
    story.append(Paragraph("• <b>Multiprocessing.Pool:</b> Uses the `starmap` method for efficient argument unpacking and process reuse.", normal))
    story.append(Paragraph("• <b>Concurrent.Futures:</b> Uses `ProcessPoolExecutor` for a modern, future-based abstraction.", normal))

    # 2. Performance Analysis
    story.append(PageBreak())
    story.append(Paragraph("2. Performance Analysis", h1))
    
    # Chart
    story.append(Paragraph("2.1 Speedup Analysis", h2))
    story.append(Image("chart_generated.png", width=6*inch, height=3.75*inch))
    
    story.append(Paragraph("<b>Observations:</b>", normal))
    story.append(Paragraph("• <b>Scalability:</b> Linear speedup is observed up to 8 workers (reaching ~6.6x).", normal))
    story.append(Paragraph("• <b>Bottlenecks:</b> At 16 workers, speedup plateaus (~7.7x). This indicates memory bandwidth saturation and IPC overhead (pickling data) outweighing the benefits of additional cores.", normal))

    # Tables
    story.append(Paragraph("2.2 Comparison Tables", h2))
    
    # Helper to build table data
    def build_table_data(strategy_name, p1_name, p2_name, dataset):
        table_data = [['Workers', 'MP Speedup', 'MP Eff', 'CF Speedup', 'CF Eff']]
        # Access data keys
        mp_key = f"{strategy_name} ({p1_name})"
        cf_key = f"{strategy_name} ({p2_name})"
        
        counts = dataset[mp_key]['workers']
        for idx, w in enumerate(counts):
            mp_s = dataset[mp_key]['speedup'][idx]
            mp_e = dataset[mp_key]['efficiency'][idx]
            cf_s = dataset[cf_key]['speedup'][idx]
            cf_e = dataset[cf_key]['efficiency'][idx]
            table_data.append([w, f"{mp_s:.2f}x", f"{mp_e:.2f}", f"{cf_s:.2f}x", f"{cf_e:.2f}"])
        return table_data

    # Table 1: Pixel Level
    story.append(Paragraph("<b>Table 1: Pixel-Level Parallelism</b>", h3))
    t1_data = build_table_data("Pixel-Level", "Multiprocessing", "Concurrent.Futures", parsed_data)
    t1 = Table(t1_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#404040")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    # Table 2: Image Level
    story.append(Paragraph("<b>Table 2: Image-Level Parallelism</b>", h3))
    t2_data = build_table_data("Image-Level", "Multiprocessing", "Concurrent.Futures", parsed_data)
    t2 = Table(t2_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#404040")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
    ]))
    story.append(t2)

    doc.build(story)
    print(f"PDF successfully generated: {filename}")

if __name__ == "__main__":
    create_pdf("Parallel_Image_Processing_Report.pdf")