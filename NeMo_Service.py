"""
NVIDIA NeMo Service - Large Language Model for content generation
"""
import httpx
import logging
from typing import Dict, List, Optional, Any
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class NeMoService:
    """Service for NVIDIA NeMo LLM operations"""
    
    def __init__(self):
        self.nemo_url = settings.NEMO_API_URL
        self.model_name = settings.NEMO_MODEL_NAME
        self.max_retries = 3
        
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using NeMo LLM
        
        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Optional stop sequences
            
        Returns:
            Generated text
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stop": stop_sequences or []
                }
                
                logger.info(f"Generating text with NeMo (temp={temperature})")
                
                response = await client.post(
                    f"{self.nemo_url}/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("text", "")
                
                logger.info(f"Generated {len(generated_text)} characters")
                return generated_text
                
        except httpx.HTTPStatusError as e:
            logger.error(f"NeMo API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}", exc_info=True)
            raise
    
    async def generate_motivational_script(
        self,
        theme: str,
        target_duration: int = 60,
        tone: str = "inspirational",
        include_call_to_action: bool = True
    ) -> Dict[str, Any]:
        """
        Generate motivational script optimized for video
        
        Args:
            theme: Theme/topic for the content
            target_duration: Target duration in seconds
            tone: Tone of the script (inspirational, energetic, calm, etc.)
            include_call_to_action: Whether to include CTA
            
        Returns:
            Dict with script, title, and metadata
        """
        # Calculate approximate word count (avg 150 words per minute)
        target_words = int((target_duration / 60) * 150)
        
        prompt = self._build_motivational_prompt(
            theme=theme,
            word_count=target_words,
            tone=tone,
            include_cta=include_call_to_action
        )
        
        try:
            generated_text = await self.generate_text(
                prompt=prompt,
                max_tokens=max(500, target_words * 2),
                temperature=0.8
            )
            
            # Parse the generated content
            script_data = self._parse_motivational_script(generated_text)
            
            return {
                "title": script_data.get("title", f"Daily Motivation: {theme}"),
                "script": script_data.get("script", generated_text),
                "theme": theme,
                "tone": tone,
                "estimated_duration": target_duration,
                "word_count": len(script_data.get("script", generated_text).split())
            }
            
        except Exception as e:
            logger.error(f"Script generation error: {str(e)}")
            raise
    
    def _build_motivational_prompt(
        self,
        theme: str,
        word_count: int,
        tone: str,
        include_cta: bool
    ) -> str:
        """Build optimized prompt for motivational content"""
        
        tone_descriptions = {
            "inspirational": "uplifting and empowering",
            "energetic": "dynamic and action-oriented",
            "calm": "soothing and reassuring",
            "professional": "authoritative and credible",
            "conversational": "friendly and relatable"
        }
        
        tone_desc = tone_descriptions.get(tone, "inspirational")
        
        prompt = f"""Create a {tone_desc} motivational script about {theme}.

Requirements:
- Length: approximately {word_count} words
- Tone: {tone}
- Include a powerful opening hook
- Provide actionable insights
- Use vivid, concrete examples
- Build emotional connection
{'- End with a clear call to action' if include_cta else ''}

Format:
TITLE: [Compelling title]

SCRIPT:
[The motivational script here]

Generate the content now:"""
        
        return prompt
    
    def _parse_motivational_script(self, generated_text: str) -> Dict[str, str]:
        """Parse generated text into structured format"""
        lines = generated_text.strip().split('\n')
        
        title = ""
        script_lines = []
        in_script = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("SCRIPT:"):
                in_script = True
            elif in_script and line:
                script_lines.append(line)
        
        script = "\n\n".join(script_lines) if script_lines else generated_text
        
        return {
            "title": title or "Daily Motivation",
            "script": script
        }
    
    async def summarize_text(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """
        Summarize long-form text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            style: Summary style (concise, detailed, bullet-points)
            
        Returns:
            Summary text
        """
        prompt = f"""Summarize the following text in a {style} style, using no more than {max_length} words:

{text[:4000]}

Summary:"""
        
        try:
            summary = await self.generate_text(
                prompt=prompt,
                max_tokens=max_length * 2,
                temperature=0.3  # Lower temperature for factual summarization
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            raise
    
    async def extract_key_insights(
        self,
        text: str,
        num_insights: int = 5
    ) -> List[str]:
        """
        Extract key insights from text
        
        Args:
            text: Source text
            num_insights: Number of insights to extract
            
        Returns:
            List of key insights
        """
        prompt = f"""Extract the {num_insights} most important and actionable insights from this text:

{text[:4000]}

Format each insight as a single, clear sentence. Number them 1-{num_insights}.

Insights:"""
        
        try:
            result = await self.generate_text(
                prompt=prompt,
                max_tokens=500,
                temperature=0.4
            )
            
            # Parse numbered list
            insights = []
            for line in result.split('\n'):
                line = line.strip()
                # Remove numbering
                for i in range(1, num_insights + 1):
                    if line.startswith(f"{i}.") or line.startswith(f"{i})"):
                        insight = line[2:].strip()
                        if insight:
                            insights.append(insight)
                        break
            
            return insights[:num_insights]
            
        except Exception as e:
            logger.error(f"Insight extraction error: {str(e)}")
            return []
    
    async def extract_themes(
        self,
        text: str,
        num_themes: int = 5
    ) -> List[str]:
        """
        Extract main themes from text
        
        Args:
            text: Source text
            num_themes: Number of themes to extract
            
        Returns:
            List of theme names
        """
        prompt = f"""Analyze this text and identify the {num_themes} main themes or topics:

{text[:4000]}

List only the theme names, one per line, without explanations.

Themes:"""
        
        try:
            result = await self.generate_text(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse themes
            themes = [
                line.strip().strip('-•*')
                for line in result.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            
            return themes[:num_themes]
            
        except Exception as e:
            logger.error(f"Theme extraction error: {str(e)}")
            return []
    
    async def generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for semantic search
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                for text in texts:
                    response = await client.post(
                        f"{self.nemo_url}/embed",
                        json={"text": text}
                    )
                    response.raise_for_status()
                    
                    embedding = response.json().get("embedding", [])
                    embeddings.append(embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]
    
    async def rewrite_for_speech(
        self,
        text: str,
        target_duration: int = 60
    ) -> str:
        """
        Rewrite text optimized for speech synthesis
        
        Args:
            text: Original text
            target_duration: Target speech duration in seconds
            
        Returns:
            Speech-optimized text
        """
        target_words = int((target_duration / 60) * 150)
        
        prompt = f"""Rewrite this text to be optimized for speech delivery. The rewritten version should:
- Be approximately {target_words} words
- Use natural, conversational language
- Avoid complex sentences
- Include appropriate pauses
- Be engaging when spoken aloud

Original text:
{text}

Speech-optimized version:"""
        
        try:
            rewritten = await self.generate_text(
                prompt=prompt,
                max_tokens=target_words * 2,
                temperature=0.7
            )
            
            return rewritten.strip()
            
        except Exception as e:
            logger.error(f"Speech rewriting error: {str(e)}")
            return text  # Return original on error
    
    async def personalize_content(
        self,
        base_content: str,
        user_preferences: Dict[str, Any]
    ) -> str:
        """
        Personalize content based on user preferences
        
        Args:
            base_content: Base motivational content
            user_preferences: User preference dict (themes, tone, etc.)
            
        Returns:
            Personalized content
        """
        pref_themes = user_preferences.get('themes', [])
        pref_tone = user_preferences.get('tone', 'inspirational')
        
        prompt = f"""Personalize this motivational content for someone interested in {', '.join(pref_themes) if pref_themes else 'general motivation'}.
Tone: {pref_tone}

Original content:
{base_content}

Personalized version:"""
        
        try:
            personalized = await self.generate_text(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            
            return personalized.strip()
            
        except Exception as e:
            logger.error(f"Personalization error: {str(e)}")
            return base_content
    
    async def batch_generate_scripts(
        self,
        themes: List[str],
        target_duration: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple scripts in batch
        
        Args:
            themes: List of themes
            target_duration: Target duration for each script
            
        Returns:
            List of script dictionaries
        """
        scripts = []
        
        for i, theme in enumerate(themes):
            try:
                logger.info(f"Generating script {i+1}/{len(themes)}: {theme}")
                
                script = await self.generate_motivational_script(
                    theme=theme,
                    target_duration=target_duration
                )
                
                scripts.append(script)
                
            except Exception as e:
                logger.error(f"Failed to generate script for theme '{theme}': {str(e)}")
                scripts.append({
                    "title": f"Error: {theme}",
                    "script": "",
                    "theme": theme,
                    "error": str(e)
                })
        
        return scripts


# Testing function
async def test_nemo_service():
    """Test NeMo service functionality"""
    service = NeMoService()
    
    # Test script generation
    script = await service.generate_motivational_script(
        theme="perseverance",
        target_duration=60,
        tone="inspirational"
    )
    
    print(f"Generated Script:")
    print(f"Title: {script['title']}")
    print(f"Script ({script['word_count']} words):")
    print(script['script'][:200] + "...")
    
    # Test theme extraction
    sample_text = """
    Success comes from hard work, dedication, and never giving up on your dreams.
    Believe in yourself and your abilities. Take action every day towards your goals.
    """
    
    themes = await service.extract_themes(sample_text, num_themes=3)
    print(f"\nExtracted themes: {themes}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_nemo_service())
