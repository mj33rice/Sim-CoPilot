import anthropic
import os
# Get the API key from environment variables
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

client = anthropic.Anthropic(
    api_key=api_key
    )

# def anthropic_models_gen(model, message, program_type, comment_symbol, max_gen_tokens):
#     message = client.messages.create(
#         model = model,
#         system = f"{program_type} code generation", 
#         max_tokens=max_gen_tokens,
#         temperature=0,
#         messages = message
#     )
#     return(message.content[0].text)

def anthropic_models_gen(model, message, program_type, comment_symbol, max_gen_tokens, enable_thinking=True, thinking_budget=16000):
    """
    Generate text using Anthropic's Claude models with optional extended thinking capability.
    
    Args:
        model (str): Anthropic model name to use
        message (list): List of message objects for the conversation
        program_type (str): Type of program for the system prompt
        comment_symbol (str): Comment symbol for the programming language
        max_gen_tokens (int): Maximum tokens for generation
        enable_thinking (bool): Whether to enable extended thinking
        thinking_budget (int): Token budget for thinking (min 1024)
    
    Notes:
        - When thinking is enabled, temperature must be 0
        - Streaming is required when max_tokens > 21,333
        - For thinking budgets > 32K, batch processing is recommended
        - Thinking is incompatible with top_p, top_k modifications and forced tool use
    """
    # Configure thinking parameters
    thinking_config = None
    if enable_thinking:
        max_gen_tokens = 20000
        thinking_config = {
            "type": "enabled",
            "budget_tokens": max(1024, thinking_budget)  # Ensure minimum 1024 tokens
        }
        
    # For very large token counts, warn about potential timeout issues
    if enable_thinking and thinking_budget > 32000:
        print(f"Warning: Thinking budget > 32K ({thinking_budget}). Consider batch processing to avoid timeouts.")
        
    # Stream for large token generations
    stream = max_gen_tokens > 21333
    # import pdb;pdb.set_trace()
    response = client.messages.create(
        model=model,
        system=f"{program_type} code generation",
        max_tokens=max_gen_tokens,
        # temperature=0,  # Required to be 0 when thinking is enabled
        messages=message,
        thinking=thinking_config,
        # stream=stream
    )
    # import pdb;pdb.set_trace()

    # Properly handle response content based on response structure
    if enable_thinking:
        # Check if thinking is available in the response
        thinking_result = getattr(response, "thinking", None)
        
        # Extract the actual text content from the response
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break
        
        # Return both thinking and response if thinking is available
        if thinking_result:
            return {
                "response": text_content,
                "thinking": thinking_result
            }
    
    # Standard response handling for non-thinking or fallback
    for block in response.content:
        if block.type == "text":
            return block.text
    
    # Last resort fallback
    return str(response.content)