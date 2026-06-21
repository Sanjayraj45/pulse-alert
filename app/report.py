import io
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.database import get_patient_history, get_risk_trend

LEVEL_COLORS = {
    'CRITICAL': colors.HexColor('#dc2626'),
    'HIGH': colors.HexColor('#ea580c'),
    'MEDIUM': colors.HexColor('#ca8a04'),
    'LOW': colors.HexColor('#16a34a')
}

def generate_trend_chart_image(trend_rows):
    """Build a risk trend line chart as PNG bytes for embedding in PDF"""
    if not trend_rows:
        return None

    scores = [round(r['risk_score'] * 100, 1) for r in trend_rows]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(range(len(scores)), scores, color='#ef4444', linewidth=2, marker='o', markersize=3)
    ax.fill_between(range(len(scores)), scores, color='#ef4444', alpha=0.08)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Risk Score (%)')
    ax.set_xlabel('Reading Number')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_patient_report(patient_id: str) -> bytes:
    """Generate a full PDF report for a patient's monitoring history"""

    history = get_patient_history(patient_id, limit=100)
    trend = get_risk_trend(patient_id, limit=100)

    if not history:
        raise ValueError(f"No data found for patient {patient_id}")

    latest = history[0]  # most recent prediction (DESC order)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.7*inch, rightMargin=0.7*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=20, spaceAfter=4)
    sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
    heading_style = ParagraphStyle('HeadStyle', parent=styles['Heading2'], fontSize=13, spaceAfter=8, spaceBefore=16)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10.5, leading=16)

    story = []

    # Header
    story.append(Paragraph("🚨 PulseAlert — Patient Risk Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}", sub_style))
    story.append(Spacer(1, 14))

    # Patient summary table
    level = latest['risk_level']
    level_color = LEVEL_COLORS.get(level, colors.black)

    summary_data = [
        ['Patient ID', patient_id],
        ['Latest Risk Score', f"{round(latest['risk_score']*100,1)}%"],
        ['Risk Level', level],
        ['Alert Status', 'ACTIVE' if latest['alert'] else 'Normal'],
        ['Total Readings Recorded', str(len(history))],
        ['Last Updated', str(latest['timestamp'])],
    ]
    t = Table(summary_data, colWidths=[2.2*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 10.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('TEXTCOLOR', (1,2), (1,2), level_color),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
    ]))
    story.append(t)

    # Clinical summary
    story.append(Paragraph("Clinical Summary", heading_style))
    nurse_summary = f"Patient shows a risk score of {round(latest['risk_score']*100,1)}% based on most recent vitals: heart rate {latest['heart_rate']} bpm, systolic BP {latest['systolic_bp']} mmHg, SpO2 {latest['spo2']}%, lactate {latest['lactate']} mmol/L, respiratory rate {latest['resp_rate']}/min, temperature {latest['temperature']}°C."
    story.append(Paragraph(nurse_summary, body_style))

    # Recommended action
    story.append(Paragraph("Recommended Action", heading_style))
    story.append(Paragraph(latest['recommended_action'], body_style))

    # Top risk factors (reading from top_factors_json, includes direction)
    story.append(Paragraph("Top Risk Factors (SHAP)", heading_style))
    factor_rows = [['Feature', 'Value', 'Impact', 'Direction']]

    factors_json = latest.get('top_factors_json')
    if factors_json:
        try:
            factors = json.loads(factors_json)
            for f in factors[:5]:
                factor_rows.append([
                    f.get('feature', '').replace('_', ' ').title(),
                    str(f.get('value', '—')),
                    f.get('impact', '—'),
                    f.get('direction', '—')
                ])
        except Exception:
            pass

    if len(factor_rows) > 1:
        ft = Table(factor_rows, colWidths=[2.3*inch, 1*inch, 1*inch, 1.4*inch])
        ft.setStyle(TableStyle([
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        story.append(ft)
    else:
        story.append(Paragraph("No factor data available for this reading.", body_style))

    # Risk trend chart
    story.append(Paragraph("Risk Score Trend Over Time", heading_style))
    chart_buf = generate_trend_chart_image(trend)
    if chart_buf:
        img = Image(chart_buf, width=6.3*inch, height=2.7*inch)
        story.append(img)
    else:
        story.append(Paragraph("Not enough data to generate trend chart.", body_style))

    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Generated by PulseAlert ICU Early Warning System. This report is for clinical decision support only and does not replace professional medical judgment.",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()