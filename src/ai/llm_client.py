"""
LLM client wrapper for the AI stuff.
Just talks to the global AI clients from meeting_processor.
"""
import logging
from typing import Optional, Dict, Any, List

# Get the global AI clients from meeting_processor
from meeting_processor import access_token, embedding_model, llm

logger = logging.getLogger(__name__)


def initialize_ai_clients():
    """
    Check if the AI clients are ready to use.
    
    This is just a wrapper - the real setup happens in meeting_processor.py
    We just check if everything is working properly.
    
    Returns:
        bool: True if everything looks good, False otherwise
    """
    try:
        logger.info("Initializing AI clients...")
        
        # Make sure the AI clients are actually loaded
        if embedding_model is None or llm is None:
            logger.error("AI clients aren't ready - check API keys")
            return False
            
        logger.info("Embedding model initialized successfully")
        logger.info("LLM initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to access AI clients: {e}")
        return False


def get_embedding_client():
    """Return the embedding model."""
    return embedding_model


def get_llm_client():
    """Return the LLM client.""" 
    return llm


def get_access_token_client():
    """Return the access token if we have one."""
    return access_token


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convert text to embedding vectors.
    
    Args:
        texts: List of text strings to convert
        
    Returns:
        List of vectors - each text becomes a list of floats
        
    Raises:
        Exception: If something goes wrong with the AI model
    """
    try:
        if embedding_model is None:
            raise Exception("Embedding model not initialized")
            
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings using the global embedding model
        embeddings = []
        for text in texts:
            if not text or not text.strip():
                # Handle empty text
                logger.warning("Empty text provided for embedding")
                embeddings.append([0.0] * 3072)  # text-embedding-3-large dimension
                continue
                
            try:
                # Call the embedding model to get vectors
                text_embeddings = embedding_model.embed_documents([text.strip()])
                embedding_vector = text_embeddings[0]
                embeddings.append(embedding_vector)
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: {e}")
                # Use empty vector if that fails
                embeddings.append([0.0] * 3072)
        
        logger.debug(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {e}")
        raise


def generate_response(prompt: str) -> Optional[str]:
    """
    Ask the AI to generate a response.
    
    Args:
        prompt: What to ask the AI
        
    Returns:
        AI response as text, or None if it fails
        
    Raises:
        Exception: If the AI model isn't working
    """
    try:
        if llm is None:
            raise Exception("LLM not initialized")
            
        logger.debug(f"Generating response for prompt (length: {len(prompt)})")
        
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided for response generation")
            return None
            
        try:
            # Send the prompt to the AI model
            response = llm.invoke(prompt.strip())
            
            # Get the actual text from the response object
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return None
                
            logger.debug(f"Successfully generated response (length: {len(response_text)})")
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise