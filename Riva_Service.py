"""
NVIDIA Riva TTS Service - Text-to-Speech synthesis
"""
import grpc
import logging
from typing import Optional, Dict, List
import numpy as np
import io

from app.core.config import settings

logger = logging.getLogger(__name__)

# Riva gRPC proto imports (these would be generated from .proto files)
# For this example, we'll use both gRPC and REST approaches

class RivaService:
    """Service for NVIDIA Riva Text-to-Speech"""
    
    def __init__(self):
        self.riva_url = settings.RIVA_API_URL
        self.default_voice = settings.RIVA_VOICE_NAME
        
    async def synthesize_speech(
        self,
        text: str,
        voice_name: Optional[str] = None,
        language_code: str = "en-US",
        sample_rate: int = 22050,
        encoding: str = "LINEAR_PCM"
    ) -> bytes:
        """
        Synthesize speech from text using NVIDIA Riva TTS
        
        Args:
            text: Text to synthesize
            voice_name: Voice name (e.g., "English-US.Female-1")
            language_code: Language code (e.g., "en-US")
            sample_rate: Audio sample rate (Hz)
            encoding: Audio encoding format
            
        Returns:
            Audio data as bytes (WAV format)
        """
        voice = voice_name or self.default_voice
        
        try:
            # Use gRPC client for Riva
            audio_data = await self._synthesize_grpc(
                text=text,
                voice=voice,
                language_code=language_code,
                sample_rate=sample_rate
            )
            
            # Convert to WAV format
            wav_data = self._create_wav_file(audio_data, sample_rate)
            
            logger.info(f"Synthesized {len(text)} characters using voice {voice}")
            return wav_data
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
            raise
    
    async def _synthesize_grpc(
        self,
        text: str,
        voice: str,
        language_code: str,
        sample_rate: int
    ) -> bytes:
        """
        Synthesize using Riva gRPC API
        
        This is a simplified implementation. In production, you would:
        1. Import generated protobuf classes
        2. Use proper Riva gRPC stubs
        3. Handle streaming responses
        """
        try:
            # Parse Riva URL
            riva_host = self.riva_url.replace('http://', '').replace('https://', '')
            
            # Create gRPC channel
            channel = grpc.aio.insecure_channel(riva_host)
            
            # In production, use actual Riva proto stubs:
            # from riva.client import SpeechSynthesisServiceStub
            # stub = SpeechSynthesisServiceStub(channel)
            
            # For now, we'll simulate with httpx as fallback
            import httpx
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                # This endpoint structure is illustrative
                # Actual Riva deployment may use gRPC or REST
                response = await client.post(
                    f"http://{riva_host}/v1/tts/synthesize",
                    json={
                        "text": text,
                        "voice": voice,
                        "language_code": language_code,
                        "sample_rate_hertz": sample_rate,
                        "encoding": "LINEAR_PCM",
                        "audio_config": {
                            "audio_encoding": "LINEAR_PCM",
                            "sample_rate_hertz": sample_rate,
                            "pitch": 0.0,
                            "speaking_rate": 1.0,
                            "volume_gain_db": 0.0
                        }
                    }
                )
                
                if response.status_code == 200:
                    # Response should contain audio bytes
                    return response.content
                else:
                    raise Exception(f"Riva TTS failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"gRPC synthesis error: {str(e)}")
            # Fallback to mock data for development
            return await self._generate_mock_audio(text, sample_rate)
    
    async def _generate_mock_audio(self, text: str, sample_rate: int) -> bytes:
        """
        Generate mock audio data for development/testing
        Creates a simple sine wave based on text length
        """
        duration = len(text) * 0.05  # ~50ms per character
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate sine wave (440 Hz tone)
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Convert to 16-bit PCM
        audio = (audio * 32767).astype(np.int16)
        
        logger.warning("Using mock audio data - Riva service not available")
        return audio.tobytes()
    
    def _create_wav_file(self, audio_data: bytes, sample_rate: int) -> bytes:
        """
        Create WAV file from PCM audio data
        
        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            
        Returns:
            Complete WAV file as bytes
        """
        import wave
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            # Set parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write audio data
            wav_file.writeframes(audio_data)
        
        # Get WAV file bytes
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    async def batch_synthesize(
        self,
        texts: List[str],
        voice_name: Optional[str] = None
    ) -> List[bytes]:
        """
        Synthesize multiple texts in batch
        
        Args:
            texts: List of texts to synthesize
            voice_name: Optional voice name
            
        Returns:
            List of audio data bytes
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                logger.info(f"Synthesizing text {i+1}/{len(texts)}")
                audio = await self.synthesize_speech(text, voice_name)
                results.append(audio)
            except Exception as e:
                logger.error(f"Failed to synthesize text {i+1}: {str(e)}")
                results.append(None)
        
        return results
    
    async def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Get list of available voices from Riva
        
        Returns:
            List of voice configurations
        """
        try:
            # In production, query Riva for available voices
            # This is a static list for example
            voices = [
                {
                    "name": "English-US.Female-1",
                    "language_code": "en-US",
                    "gender": "FEMALE",
                    "description": "US English Female Voice 1"
                },
                {
                    "name": "English-US.Male-1",
                    "language_code": "en-US",
                    "gender": "MALE",
                    "description": "US English Male Voice 1"
                },
                {
                    "name": "English-US.Female-2",
                    "language_code": "en-US",
                    "gender": "FEMALE",
                    "description": "US English Female Voice 2"
                },
                {
                    "name": "English-UK.Female-1",
                    "language_code": "en-GB",
                    "gender": "FEMALE",
                    "description": "UK English Female Voice"
                }
            ]
            
            return voices
            
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return []
    
    async def synthesize_with_ssml(
        self,
        ssml_text: str,
        voice_name: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech with SSML markup for advanced control
        
        Args:
            ssml_text: Text with SSML tags
            voice_name: Optional voice name
            
        Returns:
            Audio data bytes
        """
        # SSML allows control over:
        # - Prosody (pitch, rate, volume)
        # - Emphasis
        # - Breaks/pauses
        # - Say-as (dates, numbers, etc.)
        
        try:
            voice = voice_name or self.default_voice
            
            # Call Riva with SSML
            # In production, use proper SSML support
            audio_data = await self._synthesize_grpc(
                text=ssml_text,
                voice=voice,
                language_code="en-US",
                sample_rate=22050
            )
            
            wav_data = self._create_wav_file(audio_data, 22050)
            return wav_data
            
        except Exception as e:
            logger.error(f"SSML synthesis error: {str(e)}")
            raise
    
    async def synthesize_with_emotions(
        self,
        text: str,
        emotion: str = "neutral",
        intensity: float = 1.0,
        voice_name: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech with emotional tone
        
        Args:
            text: Text to synthesize
            emotion: Emotion type (happy, sad, excited, calm, etc.)
            intensity: Emotion intensity (0.0 to 2.0)
            voice_name: Optional voice name
            
        Returns:
            Audio data bytes
        """
        # Apply SSML tags for emotional control
        ssml_template = f"""
        <speak>
            <prosody rate="{self._get_rate_for_emotion(emotion, intensity)}"
                     pitch="{self._get_pitch_for_emotion(emotion, intensity)}"
                     volume="{self._get_volume_for_emotion(emotion, intensity)}">
                {text}
            </prosody>
        </speak>
        """
        
        return await self.synthesize_with_ssml(ssml_template, voice_name)
    
    def _get_rate_for_emotion(self, emotion: str, intensity: float) -> str:
        """Get speaking rate based on emotion"""
        rates = {
            "excited": "fast",
            "happy": "medium",
            "calm": "slow",
            "sad": "slow",
            "angry": "fast",
            "neutral": "medium"
        }
        return rates.get(emotion, "medium")
    
    def _get_pitch_for_emotion(self, emotion: str, intensity: float) -> str:
        """Get pitch based on emotion"""
        pitches = {
            "excited": "+10%",
            "happy": "+5%",
            "calm": "-5%",
            "sad": "-10%",
            "angry": "+5%",
            "neutral": "0%"
        }
        pitch = pitches.get(emotion, "0%")
        
        # Adjust for intensity
        if "+" in pitch or "-" in pitch:
            value = int(pitch.replace("%", "").replace("+", ""))
            value = int(value * intensity)
            pitch = f"{'+' if value > 0 else ''}{value}%"
        
        return pitch
    
    def _get_volume_for_emotion(self, emotion: str, intensity: float) -> str:
        """Get volume based on emotion"""
        volumes = {
            "excited": "loud",
            "happy": "medium",
            "calm": "soft",
            "sad": "soft",
            "angry": "loud",
            "neutral": "medium"
        }
        return volumes.get(emotion, "medium")


# Example usage and testing functions
async def test_riva_service():
    """Test function for Riva service"""
    service = RivaService()
    
    # Test basic synthesis
    text = "Welcome to your daily motivational message. Today is a new opportunity to grow and succeed."
    
    try:
        audio = await service.synthesize_speech(text)
        print(f"Generated audio: {len(audio)} bytes")
        
        # Test emotional synthesis
        emotional_audio = await service.synthesize_with_emotions(
            text="You can achieve anything you set your mind to!",
            emotion="excited",
            intensity=1.5
        )
        print(f"Generated emotional audio: {len(emotional_audio)} bytes")
        
        # Get available voices
        voices = await service.get_available_voices()
        print(f"Available voices: {len(voices)}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_riva_service())
