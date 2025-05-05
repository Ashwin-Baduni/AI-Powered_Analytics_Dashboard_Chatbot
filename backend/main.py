from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pydantic import BaseModel
from pathlib import Path
import json
import re
import datetime
from data_handler import AdaptiveDataParser
from calculations import Calculations
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resources - load once at startup
MODEL_PATH = "LLM/mistral-7b-openorca.Q4_K_M.gguf"
DATA_PATH = Path(__file__).parent / "newdata.json"

# Initialize resources at module level - lazy loading to improve startup time
llm = None
def get_llm():
    global llm
    if llm is None:
        llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_threads=4)
    return llm

# Response cache for common queries
response_cache = {}

# Consolidated keywords for relevance checking
KEYWORDS = {
    'vehicle_types': {
        'two_wheelers': {'bike', 'motorbike', 'motorcycle', 'scooter', 'two-wheeler'},
        'four_wheelers': {'car', 'jeep', 'suv', 'sedan', 'hatchback'},
        'commercial': {'truck', 'bus', 'lorry', 'tanker', 'excavator'}
    },
    'registration': {'registration', 'hi'},
    'statistics': {'statistics', 'data', 'count', 'total', 'average', 'increase'},
    'query_patterns': {'how many', 'count of', 'compare', 'trend of', 'forecast'},
    'attributes': {'color', 'model', 'make', 'fuel', 'engine'},
}

# Create flattened version for fast lookups
ALL_KEYWORDS = set()
for category, values in KEYWORDS.items():
    if isinstance(values, dict):
        for subcategory, keywords in values.items():
            ALL_KEYWORDS.update(keywords)
    else:
        ALL_KEYWORDS.update(values)

def is_relevant_query(query: str) -> bool:
    """
    Quickly determine if a query is related to vehicle registration data.
    """
    query_lower = query.lower()
    
    # Check for year patterns (20XX) which often indicate vehicle data queries
    if re.search(r'\b20\d{2}\b', query_lower):
        return True
    
    # Use generator expression for better performance
    return any(keyword in query_lower for keyword in ALL_KEYWORDS)

def get_irrelevant_response() -> str:
    """Return a standardized response for irrelevant queries."""
    return ("I'm a specialized assistant for vehicle registration data from Parivahan. "
            "I can help with questions about vehicle statistics, trends, or comparisons in India. "
            "Please ask me about vehicle registrations, yearly comparisons, or predictions.")

def normalize_question(text: str) -> str:
    """Normalize question text for consistent processing."""
    return re.sub(r'\s+', ' ', text.lower().strip().replace('?', ''))

def get_current_datetime():
    """Return current date and time in a formatted dictionary."""
    now = datetime.datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "weekday": now.strftime("%A"),
        "formatted": now.strftime("%A, %B %d, %Y, %H:%M %p")
    }

def get_data_year_range():
    """Get the range of years available in the data."""
    try:
        with open(DATA_PATH) as f:
            data = json.load(f)
        all_years = [int(e["year"]) for e in data.get("Original Data", []) + data.get("Predicted Data", [])]
        if all_years:
            return min(all_years), max(all_years)
    except Exception:
        pass
    return None

def dynamic_fallback_response(message: str) -> str | None:
    """Provide dynamic fallback responses for common queries."""
    msg = normalize_question(message)
    
    # Year range queries
    if any(keyword in msg for keyword in ["available years", "data available", "from which year"]):
        year_range = get_data_year_range()
        if year_range:
            min_year, max_year = year_range
            current_year = datetime.datetime.now().year
            prediction_note = ""
            if max_year > current_year:
                prediction_note = f" This includes predictions up to {max_year}."
            return f"Vehicle registration data is available from {min_year} to {max_year}.{prediction_note}"
        return "Sorry, I couldn't determine the available data range."
    
    # Category queries
    if any(term in msg for term in ["vehicle type", "category", "categories", "classification"]):
        if "available" in msg or "what" in msg:
            return "The vehicle categories available include: 2WN (Two Wheelers), LMV (Light Motor Vehicles), LPV (Light Passenger Vehicles), HGV (Heavy Goods Vehicles), LGV (Light Goods Vehicles), MGV (Medium Goods Vehicles), 3WN (Three Wheelers), HPV (Heavy Passenger Vehicles), and others. You can ask about specific categories or years."
    
    # Fuel queries
    if any(term in msg for term in ["fuel type", "fuel", "petrol", "diesel", "electric"]):
        if "available" in msg or "what" in msg:
            return "The fuel types available include: PETROL, DIESEL, CNG ONLY, LPG ONLY, ELECTRIC/PURE EV, HYBRID (including DIESEL/HYBRID, PETROL/HYBRID), SOLAR, and others. You can ask about specific fuel types or years."
    
    return None

def is_basic_query(message: str) -> bool:
    """Check if the query is a basic math expression."""
    message = message.lower()
    return (
        bool(re.match(r'^\d+[\+\-\*\/]\d+$', message)) or
        bool(re.match(r'what is \d+ (plus|minus|times|divided by) \d+\??', message))
    )

def is_simple_count_query(message: str) -> bool:
    """Check if the query is asking for a simple count of vehicles in a specific year."""
    message = message.lower()
    year_pattern = re.compile(r'(?:in|for|during)\s+(19|20)\d{2}')
    count_terms = ['how many', 'count', 'number of', 'total']
    
    has_year = bool(year_pattern.search(message))
    has_count_term = any(term in message for term in count_terms)
    
    return has_year and has_count_term and 'compare' not in message

def format_simple_count_response(context_data):
    """Format a simple count query response without using LLM."""
    if not context_data or not context_data.get('texts'):
        return None
    
    # Extract year and count information
    year_pattern = re.compile(r'In (\d{4}), there were ([\d,]+) registered vehicles')
    for text in context_data['texts']:
        match = year_pattern.search(text)
        if match:
            year, count = match.group(1), match.group(2)
            return f"In {year}, there were {count} vehicles registered."
    
    return None

def is_comparison_query(message: str) -> bool:
    """Check if the query is asking for a comparison between years."""
    message = message.lower()
    comparison_terms = ['compare', 'comparison', 'difference between', 'how does', 'versus', 'vs']
    return (
        any(term in message for term in comparison_terms) and
        len(re.findall(r'20\d{2}', message)) >= 2
    )

def format_comparison_response(context_data):
    """Format comparison data into a direct response without using LLM."""
    if not context_data or not context_data.get('texts'):
        return None
    
    # Look for comparison statements in the context
    comparison_texts = []
    for text in context_data['texts']:
        if any(term in text.lower() for term in ['increase', 'decrease', 'more than', 'less than', '%']):
            comparison_texts.append(text)
    
    # Format year data
    year_data = {}
    year_pattern = re.compile(r'In (\d{4}), there were ([\d,]+) registered vehicles')
    for text in context_data['texts']:
        match = year_pattern.search(text)
        if match:
            year = int(match.group(1))
            vehicles = int(match.group(2).replace(',', ''))
            year_data[year] = vehicles
    
    if len(year_data) >= 2:
        years = sorted(year_data.keys())
        response_parts = []
        
        # Add individual year data
        for year in years:
            response_parts.append(f"In {year}, there were {year_data[year]:,} registered vehicles.")
        
        # Add comparison between first and last years
        if len(years) >= 2:
            first_year, last_year = min(years), max(years)
            first_value = year_data[first_year]
            last_value = year_data[last_year]
            difference = last_value - first_value
            percent_change = (difference / first_value) * 100
            change_text = "increase" if difference > 0 else "decrease"
            response_parts.append(f"The number of vehicles in {last_year} is {abs(difference):,} more than in {first_year}, which represents an {change_text} of {abs(percent_change):.1f}%.")
        
        return "\n\n".join(response_parts)
    
    # If we can't create a structured response, join existing comparison texts
    if comparison_texts:
        return "\n\n".join(comparison_texts)
    
    return None

def is_prediction_query(message: str) -> bool:
    """Check if the query is asking for predictions or forecasts."""
    message = message.lower()
    prediction_terms = ['predict', 'forecast', 'future', 'next', 'projection', 'coming', 'upcoming']
    year_terms = ['year', 'years']
    
    has_prediction_term = any(term in message for term in prediction_terms)
    has_year_term = any(term in message for term in year_terms)
    
    return has_prediction_term and has_year_term

def format_prediction_response(context_data):
    """Format prediction data into a direct response without using LLM."""
    if not context_data or not context_data.get('texts'):
        return None
    
    # Look for prediction statements in the context
    prediction_texts = []
    for text in context_data['texts']:
        if "predicted" in text.lower():
            prediction_texts.append(text)
    
    if prediction_texts:
        return "\n\n".join(prediction_texts)
    
    return None

def analyze_query_complexity(query: str) -> int:
    """
    Analyze query complexity on a scale of 1-5.
    1: Simple factual query
    5: Complex analytical query requiring reasoning
    """
    query_lower = query.lower()
    
    # Simple factual queries (year counts, simple comparisons)
    if is_simple_count_query(query_lower) or is_basic_query(query_lower):
        return 1
    
    # Slightly more complex but still template-able
    if is_comparison_query(query_lower) or is_prediction_query(query_lower):
        return 2
    
    # Queries with multiple dimensions
    if ('category' in query_lower and 'fuel' in query_lower) or \
       ('ratio' in query_lower) or ('percentage' in query_lower):
        return 3
    
    # Analytical queries
    if any(term in query_lower for term in ['why', 'explain', 'analysis', 'reason']):
        return 4
    
    # Complex reasoning
    if any(term in query_lower for term in ['best', 'recommend', 'should', 'better']):
        return 5
    
    # Default to medium complexity
    return 3

def compress_context(context_texts: list, max_items: int = 10) -> str:
    """Compress context to reduce token usage."""
    if not context_texts:
        return ""
    
    # If we have many items, summarize them
    if len(context_texts) > max_items:
        # Keep the most relevant items
        relevant_items = context_texts[:max_items]
        
        # Summarize the rest
        remaining_count = len(context_texts) - max_items
        summary = f"Plus {remaining_count} more data points (omitted for brevity)."
        
        return "Here is specific vehicle registration data:\n- " + "\n- ".join(relevant_items) + f"\n- {summary}"
    
    return "Here is specific vehicle registration data:\n- " + "\n- ".join(context_texts)

def generate_template_response(query: str, context_data) -> str | None:
    """Generate template-based responses for common query patterns."""
    query_lower = query.lower()
    
    # Check for simple count queries
    if is_simple_count_query(query_lower):
        return format_simple_count_response(context_data)
    
    # Check for comparison queries
    if is_comparison_query(query_lower):
        return format_comparison_response(context_data)
    
    # Check for prediction queries
    if is_prediction_query(query_lower):
        return format_prediction_response(context_data)
    
    return None

def process_time_relative_query(query, current_datetime):
    """Process queries with relative time references like 'last year' or 'next month'."""
    query_lower = query.lower()
    
    # Map relative time terms to actual years
    current_year = current_datetime['year']
    time_mappings = {
        'last year': current_year - 1,
        'this year': current_year,
        'next year': current_year + 1,
        'previous year': current_year - 1,
        'coming year': current_year + 1,
        'two years ago': current_year - 2,
        'three years ago': current_year - 3,
        'last decade': (current_year - 10, current_year - 1)
    }
    
    # Replace relative terms with actual years
    modified_query = query_lower
    for term, year_value in time_mappings.items():
        if term in query_lower:
            if isinstance(year_value, tuple):
                modified_query = modified_query.replace(term, f"from {year_value[0]} to {year_value[1]}")
            else:
                modified_query = modified_query.replace(term, str(year_value))
    
    return modified_query if modified_query != query_lower else None

def correct_spelling(query: str) -> str:
    """Apply spelling correction to the query."""
    return AdaptiveDataParser.preprocess_query(query)

class ChatRequest(BaseModel):
    message: str
    history: list[dict]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Get current date/time
        current_datetime = get_current_datetime()
        
        # Welcome message for first interaction
        if not request.history:
            return {
                "response": f"Hi I am the Parivahan Ai Chatbot, today is {current_datetime['formatted']}. I can answer various questions like:\n- How many vehicles were there in 2008?\n- How does the number of vehicles in 2024 compare to 2004?\n- What are the predicted vehicle numbers for the next four years?\n- What is the percentage increase in vehicles from 2000 to 2024?",
                "text_context": [],
                "visual_context": []
            }
        
        # Check cache for identical queries
        cache_key = normalize_question(request.message)
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Apply spelling correction to the user's message
        corrected_message = correct_spelling(request.message)
        
        # If the message was corrected, note this for potential feedback to the user
        was_corrected = corrected_message.lower() != request.message.lower()
        correction_note = f"(Note: I interpreted '{request.message}' as '{corrected_message}')" if was_corrected else ""
        
        # Process time-relative queries
        time_processed_query = process_time_relative_query(corrected_message, current_datetime)
        query_for_processing = time_processed_query if time_processed_query else corrected_message.lower()
        
        # Early irrelevance detection - fastest path
        if not is_relevant_query(query_for_processing):
            return {
                "response": get_irrelevant_response(),
                "text_context": [],
                "visual_context": []
            }
        
        # Check for dynamic fallbacks - second fastest path
        if dynamic_response := dynamic_fallback_response(query_for_processing):
            response_obj = {
                "response": dynamic_response + ("\n\n" + correction_note if was_corrected else ""),
                "text_context": [],
                "visual_context": []
            }
            response_cache[cache_key] = response_obj
            return response_obj
        
        # Basic math - third fastest path
        if is_basic_query(corrected_message):
            result = Calculations.evaluate_expression(corrected_message)
            response_obj = {
                "response": str(result) + ("\n\n" + correction_note if was_corrected else ""),
                "text_context": [],
                "visual_context": []
            }
            response_cache[cache_key] = response_obj
            return response_obj
        
        # Get context data early for both direct answers and LLM
        context_data = AdaptiveDataParser.get_relevant_context(query_for_processing)
        
        # Simple count query - handle directly
        if is_simple_count_query(query_for_processing):
            count_response = format_simple_count_response(context_data)
            if count_response:
                response_obj = {
                    "response": count_response + ("\n\n" + correction_note if was_corrected else ""),
                    "text_context": context_data['texts'],
                    "visual_context": context_data['images']
                }
                response_cache[cache_key] = response_obj
                return response_obj
        
        # Vehicle calculations - handle common patterns directly
        if Calculations.is_calculation_query(query_for_processing):
            calculation_result = Calculations.process_query(query_for_processing)
            if calculation_result:
                response_obj = {
                    "response": calculation_result + ("\n\n" + correction_note if was_corrected else ""),
                    "text_context": context_data['texts'],
                    "visual_context": context_data['images']
                }
                response_cache[cache_key] = response_obj
                return response_obj
        
        # Ratio calculations
        ratio_result = Calculations.calculate_ratio(query_for_processing)
        if ratio_result:
            response_obj = {
                "response": ratio_result + ("\n\n" + correction_note if was_corrected else ""),
                "text_context": context_data['texts'],
                "visual_context": context_data['images']
            }
            response_cache[cache_key] = response_obj
            return response_obj
        
        # Year comparisons - handle directly without LLM
        if is_comparison_query(query_for_processing):
            comparison_response = format_comparison_response(context_data)
            if comparison_response:
                response_obj = {
                    "response": comparison_response + ("\n\n" + correction_note if was_corrected else ""),
                    "text_context": context_data['texts'],
                    "visual_context": context_data['images']
                }
                response_cache[cache_key] = response_obj
                return response_obj
        
        # Prediction queries - handle directly without LLM when possible
        if is_prediction_query(query_for_processing):
            prediction_response = format_prediction_response(context_data)
            if prediction_response:
                response_obj = {
                    "response": prediction_response + ("\n\n" + correction_note if was_corrected else ""),
                    "text_context": context_data['texts'],
                    "visual_context": context_data['images']
                }
                response_cache[cache_key] = response_obj
                return response_obj
        
        # Analyze query complexity
        complexity = analyze_query_complexity(query_for_processing)
        
        # For low complexity queries that weren't caught by specific handlers,
        # try a template-based approach
        if complexity <= 2:
            template_response = generate_template_response(query_for_processing, context_data)
            if template_response:
                response_obj = {
                    "response": template_response + ("\n\n" + correction_note if was_corrected else ""),
                    "text_context": context_data['texts'],
                    "visual_context": context_data['images']
                }
                response_cache[cache_key] = response_obj
                return response_obj
        
        # For all other queries, use the LLM with context
        # Compress context to reduce tokens
        compressed_context = compress_context(context_data['texts'])
        
        # Build prompt
        prompt = f"""You are a helpful assistant with access to vehicle registration data.
        You MUST use the provided data to answer the user's question accurately and completely.
        If the user asks about specific years or trends, refer ONLY to the data provided below.
        Do NOT make up any numbers that aren't in the provided data.
        If asked for totals or calculations, use the pre-calculated results provided.
        {compressed_context}
        
        Current date and time: {current_datetime['formatted']}
        
        User: {corrected_message}
        
        A:"""
        
        # Calculate token limits
        estimated_prompt_tokens = len(prompt) // 4
        available_tokens = min(1024, 8192 - estimated_prompt_tokens - 50)  # Reduced max_tokens
        
        # Get LLM instance (lazy loading)
        model = get_llm()
        
        # Generate response with optimized parameters
        response = model(
            prompt,
            max_tokens=available_tokens,
            temperature=0.1,
            top_p=0.95,
            echo=False,
            stop=["User:", "\n\n\n", "In conclusion", "To summarize"]
        )
        
        # Add correction note to the response if needed
        response_text = response['choices'][0]['text'].strip()
        if was_corrected:
            response_text = f"{response_text}\n\n{correction_note}"
        
        response_obj = {
            "response": response_text,
            "text_context": context_data['texts'],
            "visual_context": context_data['images']
        }
        
        # Cache the response
        response_cache[cache_key] = response_obj
        
        return response_obj
        
    except Exception as e:
        print(f"Error in chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
