import os
from google import genai
from google.genai import types

def load_markdown_content(file_path):
    """Load content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading markdown file: {e}")
        return None

def initialize_chatbot():
    """Initialize the Gemini client and setup the chatbot."""
    # Setup API client
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    # Load the markdown content
    markdown_content = load_markdown_content("combined_output.md")
    if not markdown_content:
        print("Failed to load markdown content. Exiting.")
        exit(1)
    
    # Create system prompt with instructions
    system_prompt = f"""
    You are a helpful assistant with knowledge about startup pitch decks.
    Below is a collection of text extracted from various startup pitch decks.
    Use this information to answer questions accurately.
    Only answer based on the information provided in the context.
    If the answer isn't in the context, be cretive and try to find an aswer withing the same document context, then also if you can't answer then say you don't know.
    
    CONTEXT:
    {markdown_content}
    """
    
    # Initialize chat history with system prompt
    chat_history = [
        types.Content(
            role="system",
            parts=[types.Part.from_text(text=system_prompt)],
        ),
    ]
    
    return client, chat_history

def chatbot():
    """Run the chatbot interaction loop."""
    client, chat_history = initialize_chatbot()
    model = "gemini-2.0-flash"  # You can also use "gemini-2.0-pro" for more complex reasoning
    
    print("Startup Deck Assistant initialized. Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Add user message to chat history
        chat_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )
        
        # Generate response
        try:
            print("\nAssistant: ", end="")
            
            generate_content_config = types.GenerateContentConfig(
                temperature=0.2,  # Lower temperature for factual responses
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
                response_mime_type="text/plain",
            )
            
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=chat_history,
                config=generate_content_config,
            )
            
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="")
                    full_response += chunk.text
            print()
            
            # Add model response to chat history
            chat_history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=full_response)],
                )
            )
            
            # Keep chat history to a reasonable size (optional)
            if len(chat_history) > 20:  # Adjust based on your token limits
                # Keep system prompt and last N messages
                chat_history = [chat_history[0]] + chat_history[-19:]
                
        except Exception as e:
            print(f"\nError generating response: {e}")

if __name__ == "__main__":
    chatbot()