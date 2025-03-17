import os
import streamlit as st
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import AsyncGenerator
import time

# Load environment variables
load_dotenv()

#---------------------------------------------
# RAG Backend Implementation
#---------------------------------------------
class RAGBackend:
    def __init__(self, markdown_file_path=None, markdown_content=None):
        """
        Initialize the RAG system with product data.
        Either provide a file path or markdown content directly.
        """
        # Create async OpenAI client
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
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
        
        # Ensure user_question is not None and is a string
        if user_question is None:
            user_question = ""
        else:
            user_question = str(user_question)
            
        system_prompt = self.get_system_prompt()
        
        # Ensure system_prompt is not None
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        
        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
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
        # Print debugging information
        print(f"Debug - user_question type: {type(user_question)}")
        print(f"Debug - user_question value: '{user_question}'")
        
        # Handle empty product data
        if not self.product_data:
            yield "Error: No product data available. Please check the markdown file."
            return
        
        # Force user_question to be a non-empty string
        if user_question is None or user_question == "":
            user_question = "Hello"  # Default fallback question
        
        # Ensure it's a string
        user_question = str(user_question).strip()
        
        # Double-check we have content
        if not user_question:
            user_question = "Hello"  # Another fallback

        # Get and validate system prompt
        system_prompt = self.get_system_prompt()
        if not system_prompt or system_prompt == "Error: No product data available.":
            system_prompt = "You are a helpful assistant with knowledge about startup pitch decks."
        
        # Debug the message payload
        print(f"Debug - system_prompt length: {len(system_prompt)}")
        print(f"Debug - Final user_question: '{user_question}'")
        
        # Create messages with extra validation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        try:
            # Call OpenAI API with streaming
            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                stream=True
            )
            
            # Yield each chunk as it arrives
            async for chunk in stream:
                if (hasattr(chunk.choices[0].delta, 'content') and 
                    chunk.choices[0].delta.content is not None):
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error streaming from OpenAI API: {e}")
            yield f"Error processing your request: {str(e)}"

#---------------------------------------------
# Streamlit UI Implementation
#---------------------------------------------

def load_markdown_content(file_path):
    """Load content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error loading markdown file: {e}")
        return None

def initialize_rag_backend(markdown_content):
    """Initialize the RAG backend with markdown content."""
    return RAGBackend(markdown_content=markdown_content)

def main():
    # Page config - Using centered layout with expanded sidebar
    st.set_page_config(
        page_title="Startup Deck Assistant",
        page_icon="🚀",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS for styling - UPDATED FOR WHITE THEME
    st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
            border: 1px solid #e0e0e0;
            padding: 10px 15px;
        }
        .chat-message {
            padding: 1.2rem;
            border-radius: 15px;
            margin-bottom: 1.2rem;
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        }
        .chat-message.user {
            background-color: #f0f0f0;
            border-bottom-right-radius: 5px;
            border-left: 4px solid #4e89e8;
        }
        .chat-message.assistant {
            background-color: #f8f8f8;
            border-bottom-left-radius: 5px;
            border-right: 4px solid #6c757d;
        }
        .chat-message .avatar {
            width: 45px;
            min-width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
            font-size: 22px;
            background-color: #ffffff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border: 2px solid #ffffff;
        }
        .chat-message .message {
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.6;
        }
        .button-container {
            display: flex;
            gap: 10px;
        }
        .stButton button {
            border-radius: 20px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            border: none;
        }
        .new-chat-button button {
            background-color: #4e89e8;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .new-chat-button button:hover {
            background-color: #3a76d0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .new-chat-button svg {
            margin-right: 8px;
        }
        /* Simplified title container */
        .app-title-container {
            background-color: #4e89e8;
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            text-align: center;
        }
        .app-title {
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
        }
        .app-subtitle {
            margin-top: 5px;
            font-size: 1.1rem;
        }
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 25px auto;
            text-align: center;
            width: 100%;
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #f8f9fa;
            padding-top: 2rem;
        }
        
        /* Style sidebar headings */
        .sidebar .stMarkdown h3 {
            color: #4e89e8;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .divider {
            height: 1px;
            background-color: #e0e0e0;
            margin: 25px 0;
        }
        footer {visibility: hidden;}
        .css-1dp5vir {visibility: hidden;}
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        /* Reduce padding/margins for a cleaner look */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Chat input styling */
        .stChatInput {
            padding-bottom: 10px;
        }
        .stChatInput > div > div > input::placeholder {
            color: #6c757d;
        }
        .stChatInput > div > div > input:focus {
            border-color: #4e89e8;
            box-shadow: 0 0 0 2px rgba(78, 137, 232, 0.25);
        }
        /* Status indicator styling */
        .stStatus {
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .footer-text {
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 30px;
            padding: 10px;
            border-top: 1px solid #e9ecef;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "context_loaded" not in st.session_state:
        # Load markdown content
        markdown_content = load_markdown_content("combined_output.md")
        if not markdown_content:
            st.error("Failed to load markdown content.")
            st.stop()
        
        # Store the context in session state
        st.session_state.markdown_context = markdown_content
        st.session_state.context_loaded = True
        
        # Initialize RAG backend
        st.session_state.rag_backend = initialize_rag_backend(markdown_content)
    
    # Title area - simplified layout
    st.markdown("""
    <div class="app-title-container">
        <h1 class="app-title">Startup Deck Assistant</h1>
        <p class="app-subtitle">Your personal guide to startup pitch decks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered logo display
    try:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image("image.png")

    except Exception as e:
        print(f"Error loading logo: {e}")
    
    # Add the New Chat button to sidebar
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Options</h3>", unsafe_allow_html=True)
        st.markdown('<div class="new-chat-button">', unsafe_allow_html=True)
        new_chat_button = st.button(
            "🔄 Start New Chat", 
            help="Clear current conversation and start a fresh chat",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Add additional info in the sidebar
        st.markdown("### About")
        st.markdown("This assistant helps you create and understand startup pitch decks.")
        
        if new_chat_button:
            st.session_state.messages = []
            st.session_state.processing = False
            st.success("Started a new conversation!")
            time.sleep(0.5)  # Short delay for the success message to be seen
            st.rerun()
    
    # Custom divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Display chat messages
    message_container = st.container()
    with message_container:
        if not st.session_state.messages:
            # Empty state message - REMOVED BROKEN IMAGE
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #6c757d; background-color: rgba(248,248,248,0.7); border-radius: 10px; margin: 30px 0;">
                <h3 style="margin-top: 20px; font-weight: 500;">Welcome to Startup Deck Assistant!</h3>
                <p style="margin-top: 10px; font-size: 1.1rem;">Ask me anything about startup pitch decks.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                with st.container():
                    role = message["role"]
                    content = message["content"]
                    
                    avatar = "👤" if role == "user" else "🤖"
                    background = "user" if role == "user" else "assistant"
                    
                    st.markdown(f"""
                    <div class="chat-message {background}">
                        <div class="avatar">{avatar}</div>
                        <div class="message">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Custom divider before input
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Chat input using st.chat_input instead of text_input
    user_input = st.chat_input("Ask about the startup decks...", disabled=st.session_state.processing)
    
    if user_input and not st.session_state.processing:
        # Validate user input
        print(f"Debug - Raw user input: '{user_input}'")
        user_input = str(user_input).strip()
        print(f"Debug - Processed user input: '{user_input}'")
        
        if not user_input:
            st.warning("Please enter a non-empty message.")
            st.stop()
        
        # Mark as processing to prevent multiple requests
        st.session_state.processing = True
        
        # Add user message to UI chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Rerun to show user message immediately
        st.rerun()
    
    # Process the message after showing user message
    if st.session_state.processing and len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # Get the last user message
        user_message = st.session_state.messages[-1]["content"]
        print(f"Debug - Processing user message: '{user_message}'")
        
        # Extra validation to ensure we have a valid user message
        if not user_message or user_message.strip() == "":
            st.error("Cannot process empty message")
            st.session_state.processing = False
            st.rerun()
        
        # Create status indicator
        status = st.status("Generating response...", expanded=False)
        
        # Create a placeholder for the assistant's response
        with st.container():
            assistant_msg = {"role": "assistant", "content": ""}
            st.session_state.messages.append(assistant_msg)
            
            # Reference to update the last message
            last_idx = len(st.session_state.messages) - 1
            
            message_placeholder = st.empty()
            message_placeholder.markdown(f"""
            <div class="chat-message assistant">
                <div class="avatar">🤖</div>
                <div class="message">Thinking...</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Generate response using the RAG backend with validated user message
                async def run_stream_query():
                    full_response = ""
                    try:
                        async for text_chunk in st.session_state.rag_backend.stream_query(user_message):
                            if text_chunk:  # Check if chunk is not empty
                                full_response += text_chunk
                                message_placeholder.markdown(f"""
                                <div class="chat-message assistant">
                                    <div class="avatar">🤖</div>
                                    <div class="message">{full_response}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Update the message in session state
                                st.session_state.messages[last_idx]["content"] = full_response
                                status.update(label="Generating response...", state="running")
                                time.sleep(0.01)  # Slight delay for UI updates
                        
                        # If we got an empty response, provide a fallback
                        if not full_response:
                            full_response = "I'm having trouble generating a response right now. Please try again."
                            message_placeholder.markdown(f"""
                            <div class="chat-message assistant">
                                <div class="avatar">🤖</div>
                                <div class="message">{full_response}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.messages[last_idx]["content"] = full_response
                            
                    except Exception as e:
                        print(f"Stream query error: {e}")
                        full_response = f"Sorry, there was an error generating a response: {str(e)}"
                        message_placeholder.markdown(f"""
                        <div class="chat-message assistant">
                            <div class="avatar">🤖</div>
                            <div class="message">{full_response}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.messages[last_idx]["content"] = full_response
                    
                    # Final update with complete response
                    status.update(label="Response complete!", state="complete")
                
                # Run the async function
                asyncio.run(run_stream_query())
                
                # Close status after a short delay
                time.sleep(0.5)
                status.update(state="complete", expanded=False)
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.markdown(f"""
                <div class="chat-message assistant">
                    <div class="avatar">🤖</div>
                    <div class="message">{error_msg}</div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.messages[last_idx]["content"] = error_msg
                status.update(label="Error", state="error")
            
            # Reset processing flag
            st.session_state.processing = False
            st.rerun()

    # Footer
    st.markdown("""
    <div class="footer-text">
        <p>Startup Deck Assistant - Powered by OpenAI & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()