from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io
from PIL import Image
import base64

class PainAssessmentReport:
    """
    Generate comprehensive PDF reports for pain assessments
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            name='PatientInfo',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        ))
    
    def create_pain_scale_reference(self):
        """Create pain scale reference table"""
        scale_data = [
            ['PSPI Score', 'Pain Level', 'Clinical Description'],
            ['0-1', 'No Pain', 'No visible signs of discomfort'],
            ['1-3', 'Minimal Pain', 'Slight facial tension, barely noticeable'],
            ['3-5', 'Mild Pain', 'Noticeable facial expression changes'],
            ['5-7', 'Moderate Pain', 'Clear pain indicators, frowning, tension'],
            ['7-10', 'Severe Pain', 'Significant facial distortion, eye closing'],
            ['10+', 'Very Severe', 'Extreme facial expression, maximum distress']
        ]
        
        table = Table(scale_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def generate_report(self, patient_data: dict, reference_image: bytes, 
                       target_image: bytes, pain_score: float, 
                       output_path: str = None) -> bytes:
        """
        Generate comprehensive pain assessment PDF report
        
        Args:
            patient_data: Dict containing patient information
            reference_image: Reference image bytes
            target_image: Target image bytes  
            pain_score: Calculated pain score
            output_path: Optional file path to save PDF
            
        Returns:
            PDF bytes
        """
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title
        title = Paragraph("SCALA DOLORE - Pain Assessment Report", 
                         self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Patient Information Table
        patient_info_data = [
            ['Patient Information', ''],
            ['Patient ID:', patient_data.get('patient_id', 'N/A')],
            ['Name:', patient_data.get('name', 'N/A')],
            ['Date of Birth:', patient_data.get('dob', 'N/A')],
            ['Assessment Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Doctor:', patient_data.get('doctor', 'N/A')],
        ]
        
        patient_table = Table(patient_info_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Assessment Results
        pain_level = self.get_pain_level_description(pain_score)
        results_data = [
            ['Assessment Results', ''],
            ['PSPI Pain Score:', f"{pain_score:.2f}"],
            ['Pain Level:', pain_level],
            ['Assessment Method:', 'Facial Expression Analysis (SCALA DOLORE)'],
            ['AI Model:', 'ConvNetOrdinalLateFusion with Contrastive Learning'],
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Images Comparison Section
        story.append(Paragraph("Image Comparison Analysis", 
                              self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Convert images for PDF
        ref_img = self.prepare_image_for_pdf(reference_image, "Reference Image")
        target_img = self.prepare_image_for_pdf(target_image, "Current Assessment")
        
        # Image comparison table
        images_data = [
            ['Reference Image (Baseline)', 'Current Assessment Image'],
            [ref_img, target_img],
            ['Patient baseline expression', f'Current expression (Score: {pain_score:.2f})']
        ]
        
        images_table = Table(images_data, colWidths=[3*inch, 3*inch])
        images_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(images_table)
        story.append(Spacer(1, 20))
        
        # Pain Scale Reference
        story.append(Paragraph("PSPI Pain Scale Reference", 
                              self.styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(self.create_pain_scale_reference())
        story.append(Spacer(1, 20))
        
        # Clinical Notes Section
        story.append(Paragraph("Clinical Notes", self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        clinical_notes = f"""
        This assessment was performed using the SCALA DOLORE system, which employs 
        advanced facial expression analysis to objectively measure pain intensity. 
        The system compares the current facial expression against the patient's 
        baseline (reference) image to detect changes indicative of pain.
        
        Current Assessment: {pain_level}
        Recommendation: {self.get_clinical_recommendation(pain_score)}
        
        Note: This automated assessment should be used in conjunction with clinical 
        judgment and patient self-reporting when possible.
        """
        
        story.append(Paragraph(clinical_notes, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        return pdf_bytes
    
    def prepare_image_for_pdf(self, image_bytes: bytes, caption: str):
        """Convert image bytes to ReportLab Image object"""
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return RLImage(img_buffer, width=2*inch, height=2*inch)
    
    def get_pain_level_description(self, score: float) -> str:
        """Get pain level description based on PSPI score"""
        if score < 1:
            return "No Pain"
        elif score < 3:
            return "Minimal Pain"
        elif score < 5:
            return "Mild Pain"
        elif score < 7:
            return "Moderate Pain"
        elif score < 10:
            return "Severe Pain"
        else:
            return "Very Severe Pain"
    
    def get_clinical_recommendation(self, score: float) -> str:
        """Get clinical recommendation based on pain score"""
        if score < 3:
            return "Continue monitoring. No immediate intervention required."
        elif score < 5:
            return "Consider non-pharmacological interventions."
        elif score < 7:
            return "Evaluate for pain management interventions."
        else:
            return "Immediate pain management assessment recommended."