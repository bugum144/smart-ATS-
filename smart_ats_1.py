import os
import re
import time
import json
import asyncio
import pandas as pd
from io import BytesIO
from typing import Optional, Dict
import streamlit as st
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

class ResumeAnalyzer:
    def __init__(self):
        """Initialize ResumeAnalyzer with optimized settings."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cached_responses: Dict[str, str] = {}
        self.profiling_data = {
            "text_extraction": [],
            "ai_processing": [],
            "pdf_generation": []
        }

    def configure_generative_ai(self) -> bool:
        """Configure Generative AI model with error handling."""
        if not self.api_key:
            st.error("Google API key not found. Please set GOOGLE_API_KEY.")
            return False
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            return True
        except Exception as e:
            st.error(f"AI configuration failed: {str(e)}")
            return False

    async def extract_text_async(self, file) -> Optional[str]:
        """Async text extraction with format detection."""
        loop = asyncio.get_running_loop()
        try:
            if file.name.lower().endswith('.pdf'):
                return await loop.run_in_executor(
                    self.executor, 
                    self._extract_text_from_pdf, 
                    file
                )
            elif file.name.lower().endswith('.docx'):
                return await loop.run_in_executor(
                    self.executor,
                    self._extract_text_from_docx,
                    file
                )
            else:
                st.error("Unsupported file type. Please upload PDF or DOCX.")
                return None
        except Exception as e:
            st.error(f"Text extraction error: {str(e)}")
            return None

    def _extract_text_from_pdf(self, file) -> Optional[str]:
        """Improved PDF extraction using pdfplumber."""
        start_time = time.time()
        try:
            text = []
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
            
            extracted_text = ' '.join(text)
            self.profiling_data["text_extraction"].append(time.time() - start_time)
            return extracted_text
        except Exception as e:
            st.error(f"PDF extraction failed: {str(e)}")
            return None

    def _extract_text_from_docx(self, file) -> Optional[str]:
        """DOCX text extraction."""
        start_time = time.time()
        try:
            doc = Document(file)
            text = [p.text for p in doc.paragraphs]
            extracted_text = '\n'.join(text)
            self.profiling_data["text_extraction"].append(time.time() - start_time)
            return extracted_text
        except Exception as e:
            st.error(f"DOCX extraction failed: {str(e)}")
            return None

    async def get_gemini_response(self, prompt: str, use_cache: bool = True) -> Optional[str]:
        """Get cached or new AI response with error handling."""
        if use_cache and prompt in self.cached_responses:
            return self.cached_responses[prompt]
            
        if not self.model:
            st.error("AI model not initialized")
            return None
            
        try:
            start_time = time.time()
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            self.profiling_data["ai_processing"].append(time.time() - start_time)
            
            if response.text:
                self.cached_responses[prompt] = response.text
                return response.text
            return None
        except Exception as e:
            st.error(f"AI request failed: {str(e)}")
            return None

    async def analyze_resume(self, job_description: str, resume_text: str) -> Optional[Dict]:
        """Enhanced resume analysis with professional output format."""
        analysis_prompt = f"""
        Analyze this resume against the job description and return a comprehensive professional report in this exact format:

        ### Candidate Matching Score: {{percentage_match}}%
        - Job Title: {{job_title}}
        - Resume Analysis: {{analysis_summary}}

        ### Highlighted Keywords Comparison
        | Category          | Job Keyword   | Missing Keyword | Resume Keyword | Percentage Match |
        |-------------------|---------------|-----------------|----------------|------------------|
        {{keyword_table}}

        ### Missing Keywords
        {{missing_keywords_list}}

        ### Educational Resources
        {{educational_resources}}

        ### Optimization Suggestions
        {{optimization_suggestions}}

        Notes:
        - Be professional and constructive
        - Provide actionable recommendations
        - Include relevant learning resources

        Job Description: {job_description[:3000]}
        Resume: {resume_text[:3000]}
        """
        
        response = await self.get_gemini_response(analysis_prompt)
        return self._parse_analysis_response(response)

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse the structured analysis response into components."""
        result = {
            "score": 0,
            "job_title": "",
            "analysis_summary": "",
            "keyword_table": "",
            "missing_keywords": [],
            "resources": "",
            "suggestions": ""
        }
        
        if not response:
            return result
        
        # Extract score
        score_match = re.search(r"### Candidate Matching Score: (\d+)%", response)
        if score_match:
            result["score"] = int(score_match.group(1))
        
        # Extract job title
        title_match = re.search(r"- Job Title: (.+)", response)
        if title_match:
            result["job_title"] = title_match.group(1).strip()
        
        # Extract analysis summary
        analysis_match = re.search(r"- Resume Analysis: (.+?)(?=###)", response, re.DOTALL)
        if analysis_match:
            result["analysis_summary"] = analysis_match.group(1).strip()
        
        # Extract keyword table
        table_match = re.search(r"### Highlighted Keywords Comparison(.+?)(?=###)", response, re.DOTALL)
        if table_match:
            result["keyword_table"] = table_match.group(1).strip()
        
        # Extract missing keywords
        missing_match = re.search(r"### Missing Keywords(.+?)(?=###)", response, re.DOTALL)
        if missing_match:
            result["missing_keywords"] = missing_match.group(1).strip()
        
        # Extract educational resources
        resources_match = re.search(r"### Educational Resources(.+?)(?=###)", response, re.DOTALL)
        if resources_match:
            result["resources"] = resources_match.group(1).strip()
        
        # Extract suggestions
        suggestions_match = re.search(r"### Optimization Suggestions(.+?)(?=Notes:)", response, re.DOTALL)
        if suggestions_match:
            result["suggestions"] = suggestions_match.group(1).strip()
        
        return result

    def display_analysis_results(self, analysis: Dict):
        """Display the analysis results in Streamlit with enhanced formatting."""
        st.subheader("📊 Resume Analysis Results")
        
        # Score and Overview
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Candidate Matching Score", f"{analysis.get('score', 0)}%")
        with col2:
            if analysis.get('job_title'):
                st.metric("Job Title", analysis['job_title'])
        
        st.divider()
        
        # Analysis Summary
        if analysis.get('analysis_summary'):
            with st.expander("📝 Analysis Summary"):
                st.write(analysis['analysis_summary'])
        
        # Keyword Analysis
        if analysis.get('keyword_table'):
            st.subheader("🔍 Keyword Comparison")
            table_lines = [
                line.split("|")[1:-1] 
                for line in analysis['keyword_table'].split("\n") 
                if line.startswith("|")
            ]
            if len(table_lines) > 1:
                headers = [h.strip() for h in table_lines[0]]
                data = [
                    [d.strip() for d in row] 
                    for row in table_lines[1:] 
                    if any(x.strip() for x in row)
                ]
                st.dataframe(
                    pd.DataFrame(data, columns=headers),
                    use_container_width=True,
                    hide_index=True
                )
        
        # Missing Keywords
        if analysis.get('missing_keywords'):
            st.subheader("⚠️ Missing Keywords")
            st.markdown(analysis['missing_keywords'])
        
        # Educational Resources
        if analysis.get('resources'):
            st.subheader("📚 Recommended Learning Resources")
            st.markdown(analysis['resources'])
        
        # Optimization Suggestions
        if analysis.get('suggestions'):
            st.subheader("💡 Optimization Suggestions")
            st.markdown(analysis['suggestions'])

    async def generate_optimized_resume(self, job_description: str, resume_text: str, analysis: Dict) -> Optional[str]:
        """Generate optimized resume using analysis results."""
        optimization_prompt = f"""
        Optimize this resume using the analysis below.
        Keep original structure but enhance with missing keywords.
        Use concise bullet points (max 4 per role).
        Return in this format:
        
        [CONTACT INFO]
        [EDUCATION]
        [SKILLS] 
        [EXPERIENCE]
        
        Original Resume: {resume_text}
        Job Description: {job_description}
        Analysis: {json.dumps(analysis)}
        """
        
        return await self.get_gemini_response(optimization_prompt)

    def generate_formatted_pdf(self, text: str) -> Optional[BytesIO]:
        """Improved PDF generation with templates."""
        start_time = time.time()
        try:
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # Header
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 800, "Optimized Resume")
            c.line(50, 795, 550, 795)
            
            # Body
            c.setFont("Helvetica", 12)
            y_position = 750
            for line in text.split('\n'):
                if y_position < 50:
                    c.showPage()
                    y_position = 750
                c.drawString(50, y_position, line)
                y_position -= 15
                
            c.save()
            buffer.seek(0)
            self.profiling_data["pdf_generation"].append(time.time() - start_time)
            return buffer
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")
            return None

    def get_performance_metrics(self) -> Dict:
        """Return average processing times."""
        return {
            "text_extraction": self._safe_avg(self.profiling_data["text_extraction"]),
            "ai_processing": self._safe_avg(self.profiling_data["ai_processing"]),
            "pdf_generation": self._safe_avg(self.profiling_data["pdf_generation"])
        }

    def _safe_avg(self, values: list) -> float:
        """Calculate average safely for empty lists."""
        return sum(values)/len(values) if values else 0

async def main():
    """Main async application flow."""
    st.set_page_config(
        page_title="Smart ATS Resume Optimizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize analyzer instance at the start
    analyzer = ResumeAnalyzer()
    
    # Initialize session state
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'optimized_resume' not in st.session_state:
        st.session_state.optimized_resume = None
    
    # UI Setup
    st.title("📄 Smart ATS Resume Optimizer")
    st.caption("Improve your resume's ATS compatibility")
    
    with st.sidebar:
        st.header("Settings")
        job_description = st.text_area(
            "Paste Job Description",
            height=200,
            help="The job description you're targeting"
        )
        
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx"],
            help="PDF or DOCX files only"
        )
        
        dark_mode = st.toggle("Dark Mode", value=False)
        st.session_state.dark_mode = dark_mode
        
        if st.button("🔍 Analyze Resume", use_container_width=True):
            if uploaded_file and job_description:
                with st.spinner("Extracting text..."):
                    if analyzer.configure_generative_ai():
                        st.session_state.resume_text = await analyzer.extract_text_async(uploaded_file)
                        
                        if st.session_state.resume_text:
                            with st.spinner("Analyzing..."):
                                st.session_state.analysis = await analyzer.analyze_resume(
                                    job_description,
                                    st.session_state.resume_text
                                )
            else:
                st.error("Please upload a resume and provide a job description")
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["Analysis", "Optimized Resume", "Performance"])
    
    with tab1:
        if st.session_state.analysis:
            analyzer.display_analysis_results(st.session_state.analysis)
        else:
            st.info("Upload a resume and click Analyze to get started")

    with tab2:
        if st.session_state.analysis and st.session_state.resume_text:
            if st.button("✨ Generate Optimized Resume"):
                with st.spinner("Generating optimized resume..."):
                    if analyzer.configure_generative_ai():
                        st.session_state.optimized_resume = await analyzer.generate_optimized_resume(
                            job_description,
                            st.session_state.resume_text,
                            st.session_state.analysis
                        )
            
            if st.session_state.optimized_resume:
                st.subheader("Optimized Resume")
                with st.expander("View Resume Content"):
                    st.write(st.session_state.optimized_resume)
                
                # PDF Download
                pdf_buffer = analyzer.generate_formatted_pdf(st.session_state.optimized_resume)
                if pdf_buffer:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_buffer,
                        file_name="optimized_resume.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("Complete analysis first to generate optimized resume")

    with tab3:
        metrics = analyzer.get_performance_metrics()
        st.subheader("⏱️ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Text Extraction", f"{metrics['text_extraction']:.2f}s")
        with col2:
            st.metric("AI Processing", f"{metrics['ai_processing']:.2f}s")
        with col3:
            st.metric("PDF Generation", f"{metrics['pdf_generation']:.2f}s")
        
        # Safe progress bar calculation
        progress_value = min(int(metrics['ai_processing'] * 10), 100)
        st.progress(
            progress_value, 
            text=f"AI Processing Efficiency: {progress_value}%"
        )
        
if __name__ == "__main__":
    asyncio.run(main())