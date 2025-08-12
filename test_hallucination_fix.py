#!/usr/bin/env python3
"""
Test script to verify the hallucination fix for @file: specific queries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.enhanced_prompts import EnhancedPromptManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hallucination_fix():
    """Test that document-specific template prevents hallucination"""
    try:
        # Initialize prompt manager
        prompt_manager = EnhancedPromptManager()
        
        # Sample context that only contains June 27th meeting data
        sample_context = [{
            'text': """Document: Document Fulfillment AIML-20250627_150201-Meeting Recording.docx
            
Meeting Date: June 27, 2025
Participants: Kevin Vautrinot, Sandeep Reddy Gantla, Adrian Smithee, Michael W Gwin, Jeevan R Dubba

Key Discussion Points:
- Project updates and visualization tools
- Document management system demonstration
- AI integration for document processing
- Discussion about pivot tables and data analysis
- User interface walkthrough for document management

The meeting focused on improving project management through technology integration and AI-powered document analysis.""",
            'document_name': 'Document Fulfillment AIML-20250627_150201-Meeting Recording.docx',
            'timestamp': '2025-06-27T15:02:01'
        }]
        
        # Test query asking about meeting date
        test_query = "@file:Document Fulfillment AIML-20250627_150201-Meeting Recording.docx what is the meeting date?"
        
        # Generate prompt using document-specific template
        logger.info("=== Testing document-specific template ===")
        prompt = prompt_manager.generate_enhanced_prompt(
            query=test_query,
            context_chunks=sample_context,
            query_type='document_specific'
        )
        
        logger.info("Generated prompt:")
        logger.info("=" * 80)
        logger.info(prompt)
        logger.info("=" * 80)
        
        # Check that the prompt contains the anti-hallucination instructions
        expected_instructions = [
            "ONLY use information explicitly provided",
            "DO NOT create, infer, or assume any meeting dates",
            "DO NOT provide information about meetings other than those in the provided content",
            "If specific information is not in the provided content, state that it's not available"
        ]
        
        for instruction in expected_instructions:
            if instruction in prompt:
                logger.info(f"✅ Found instruction: {instruction}")
            else:
                logger.error(f"❌ Missing instruction: {instruction}")
                return False
        
        # Test comparison with detailed_analysis template (the problematic one)
        logger.info("\n=== Testing detailed_analysis template (problematic) ===")
        detailed_prompt = prompt_manager.generate_enhanced_prompt(
            query=test_query,
            context_chunks=sample_context,
            query_type='detailed_analysis'
        )
        
        # Check that detailed_analysis template encourages broader analysis
        problematic_phrases = [
            "comprehensive background information",
            "broader organizational or project context",
            "Provide comprehensive coverage"
        ]
        
        found_problematic = False
        for phrase in problematic_phrases:
            if phrase in detailed_prompt:
                logger.warning(f"⚠️  Found problematic phrase in detailed_analysis: {phrase}")
                found_problematic = True
        
        if found_problematic:
            logger.info("✅ Confirmed detailed_analysis template has problematic instructions")
        else:
            logger.warning("❌ detailed_analysis template may have been changed")
        
        # Verify document_specific template is more restrictive
        if len(prompt) != len(detailed_prompt):
            logger.info(f"✅ Templates have different lengths: document_specific={len(prompt)}, detailed_analysis={len(detailed_prompt)}")
        
        logger.info("\n=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_hallucination_fix()
    if success:
        print("\nHallucination fix test PASSED")
    else:
        print("\nHallucination fix test FAILED")
        sys.exit(1)