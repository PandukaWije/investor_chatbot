import os
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()


# Create async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGBackend:
    def __init__(self, markdown_file_path=None, markdown_content=None):
        """
        Initialize the RAG system with product data.
        Either provide a file path or markdown content directly.
        """
        self.markdown_file_path = markdown_file_path
        if markdown_content:
            self.product_data = markdown_content
        elif markdown_file_path:
            self.product_data = self._load_markdown_file()
        else:
            self.product_data = ""
    
    def _load_markdown_file(self):
        """Load and read the markdown file."""
        try:
            with open(self.markdown_file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading markdown file: {e}")
            return ""
    
    def get_system_prompt(self, user_question=None):
        """Generate the system prompt with product data and instructions."""
        if not self.product_data:
            return "Error: No product data available."
        
        return f"""
        You are a helpful assistant with knowledge about startup pitch decks.
        The information below is extracted from various startup pitch decks.
        
        {self.product_data}
          
        Use this information to answer questions accurately.
        Only answer based on the information provided in the context.
        If the answer isn't in the context, be creative and try to find an answer within the same document context, then also if you can't answer then say you don't know. 
        """
    
    async def query(self, user_question):
        """
        Query the product information based on user question.
        Uses OpenAI API to generate a response based on the product data.
        """
        if not self.product_data:
            return "Error: No product data available. Please check the markdown file."
        
        system_prompt = self.get_system_prompt()
        
        try:
            # Call OpenAI API
            response = await client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.1  # Lower temperature for more factual responses
            )
            
            # Return the assistant's response
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error querying OpenAI API: {e}")
            return f"Error processing your request: {str(e)}"
    
    async def stream_query(self, user_question) -> AsyncGenerator[str, None]:
        """
        Stream the response from OpenAI API for a given user question.
        """
        if not self.product_data:
            yield "Error: No product data available. Please check the markdown file."
            return
        
        system_prompt = self.get_system_prompt()
        
        try:
            # Call OpenAI API with streaming
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.1,
                stream=True
            )
            
            # Yield each chunk as it arrives
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error streaming from OpenAI API: {e}")
            yield f"Error processing your request: {str(e)}"
