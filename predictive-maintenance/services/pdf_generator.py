"""
PDF Report Generator for Maintenance Intelligence Reports
Converts markdown reports to professionally formatted PDFs
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import io
from typing import Dict, Any


class PDFReportGenerator:
    """Generate professional maintenance reports as PDFs"""
    
    def __init__(self):
        """Initialize PDF generator with custom styles"""
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Define custom paragraph styles for the report"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a5f'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4a9eff'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#2d5a7b'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles['BodyText'].fontSize = 10
        self.styles['BodyText'].leading = 14
        self.styles['BodyText'].textColor = colors.HexColor('#2d2d2d')
        self.styles['BodyText'].spaceAfter = 8
        
        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#7a7a7a'),
            alignment=TA_RIGHT,
            spaceAfter=20
        ))
    
    def _add_header(self, canvas, doc):
        """Add header to each page"""
        canvas.saveState()
        
        # Header background
        canvas.setFillColor(colors.HexColor('#1e3a5f'))
        canvas.rect(0, letter[1] - 0.8*inch, letter[0], 0.8*inch, fill=True, stroke=False)
        
        # Header text
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(0.75*inch, letter[1] - 0.5*inch, "⚙️ Industrial Maintenance Intelligence")
        
        # Page number
        canvas.setFont('Helvetica', 9)
        canvas.drawRightString(letter[0] - 0.75*inch, 0.5*inch, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _create_health_score_box(self, health_score: float, risk_level: str) -> Table:
        """Create a colored box showing health score and risk level"""
        
        # Determine color based on risk level
        color_map = {
            'Critical': colors.HexColor('#ef4444'),
            'High': colors.HexColor('#ef9944'),
            'Medium': colors.HexColor('#fbbf24'),
            'Low': colors.HexColor('#34d399')
        }
        box_color = color_map.get(risk_level, colors.grey)
        
        # Create table data
        data = [
            [Paragraph(f"<b>Health Score: {health_score:.1f}/100</b>", self.styles['BodyText'])],
            [Paragraph(f"<b>Risk Level: {risk_level}</b>", self.styles['BodyText'])]
        ]
        
        # Style the table
        table = Table(data, colWidths=[3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), box_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 2, colors.white)
        ]))
        
        return table
    
    def _parse_report_sections(self, report_text: str) -> Dict[str, str]:
        """Parse markdown report into sections"""
        sections = {}
        current_section = None
        current_content = []
        
        lines = report_text.split('\n')
        
        for line in lines:
            # Detect section headers (## Section Name)
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.replace('##', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def generate_pdf(
        self,
        report_text: str,
        machine_id: str,
        health_score: float,
        risk_level: str
    ) -> bytes:
        """
        Generate PDF from markdown report
        
        Args:
            report_text: Markdown formatted report
            machine_id: Machine identifier
            health_score: Health score (0-100)
            risk_level: Risk level (Critical/High/Medium/Low)
        
        Returns:
            PDF file as bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Build story (content)
        story = []
        
        # Title
        story.append(Paragraph("MAINTENANCE INTELLIGENCE REPORT", self.styles['ReportTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata
        metadata = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} | Asset: {machine_id}"
        story.append(Paragraph(metadata, self.styles['Metadata']))
        story.append(Spacer(1, 0.1*inch))
        
        # Health score box
        story.append(self._create_health_score_box(health_score, risk_level))
        story.append(Spacer(1, 0.3*inch))
        
        # Parse and add sections
        sections = self._parse_report_sections(report_text)
        
        for section_title, section_content in sections.items():
            # Skip the title section (already added)
            if 'MAINTENANCE INTELLIGENCE REPORT' in section_title:
                continue
            
            # Add section header
            story.append(Paragraph(section_title, self.styles['SectionHeader']))
            
            # Process section content
            paragraphs = section_content.split('\n\n')
            
            for para in paragraphs:
                if not para.strip():
                    continue
                
                # Handle bold markdown
                #para = para.replace('**', '<b>').replace('**', '</b>')
                
                # Handle bullet points
                if para.strip().startswith('- '):
                    # Convert markdown bullets to HTML
                    bullets = para.split('\n- ')
                    for bullet in bullets:
                        if bullet.strip():
                            clean_bullet = bullet.replace('- ', '').strip()
                            story.append(Paragraph(f"• {clean_bullet}", self.styles['BodyText']))
                else:
                    # Regular paragraph
                    story.append(Paragraph(para, self.styles['BodyText']))
                
                story.append(Spacer(1, 0.1*inch))
        
        # Footer note
        story.append(Spacer(1, 0.3*inch))
        footer_text = "<i>This report is generated by AI-powered maintenance intelligence systems and should be reviewed by qualified maintenance personnel before taking action.</i>"
        story.append(Paragraph(footer_text, self.styles['BodyText']))
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_header, onLaterPages=self._add_header)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes


# Convenience function
def generate_maintenance_pdf(
    report_text: str,
    machine_id: str,
    health_score: float,
    risk_level: str
) -> bytes:
    """
    Generate maintenance report PDF
    
    Args:
        report_text: Full markdown report text
        machine_id: Machine identifier
        health_score: Health score (0-100)
        risk_level: Risk level string
    
    Returns:
        PDF file as bytes
    """
    generator = PDFReportGenerator()
    return generator.generate_pdf(report_text, machine_id, health_score, risk_level)