import streamlit as st
import base64
import json
import os
from typing import List, Set, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="StyleSync - AI Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pydantic Models for structured data extraction
class ClothingItem(BaseModel):
    category: str = Field(..., description="Category of the clothing item (e.g., T-Shirt, Dress, Pants, Shorts)")
    description: str = Field(..., description="Detailed description of the clothing item")
    color: List[str] = Field(..., description="Available colors of the clothing item")
    gender: str = Field(..., description="Gender suitability (Unisex, Male, Female)")
    fabric: str = Field(..., description="Type of fabric used")
    pattern: str = Field(..., description="Pattern (Solid, Striped, Checked, Floral, etc.)")
    fit: str = Field(..., description="Fit type (Regular Fit, Slim Fit, Loose Fit)")
    sleeve_length: str = Field(..., description="Sleeve length (Short, Long, 3/4, Sleeveless, N/A)")
    neck_type: str = Field(..., description="Neck type (Round, V-Neck, Collar, etc.)")
    occasion: List[str] = Field(..., description="Suitable occasions")
    season: List[str] = Field(..., description="Suitable seasons")
    features: Set[str] = Field(..., description="Special features")

class OutfitRecommendation(BaseModel):
    recommended_items: List[str] = Field(..., description="List of clothing item IDs for the recommended outfit")
    reasoning: str = Field(..., description="Explanation for why this outfit was recommended")
    style_tips: List[str] = Field(..., description="Additional styling tips")

class StyleSyncBot:
    def __init__(self, api_key: str):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
                api_key=api_key,
                max_tokens=None,
                timeout=60,
                max_retries=3,
            )
            self.structured_llm = self.llm.with_structured_output(ClothingItem, method="json-mode")
            self.recommendation_llm = self.llm.with_structured_output(OutfitRecommendation, method="json-mode")
            self.wardrobe = []
        except Exception as e:
            st.error(f"Failed to initialize AI model: {str(e)}")
            raise e

    def encode_image(self, image_bytes):
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode("utf-8")

    def analyze_clothing_image(self, image_bytes):
        """Analyze uploaded clothing image and extract structured data"""
        try:
            image_base64 = self.encode_image(image_bytes)
            
            prompt = [
                SystemMessage(
                    content=[
                        {"type": "text", "text": """You are an expert fashion analyst. Analyze the clothing item in the image and extract detailed information about its properties. 
                        Focus on identifying the category, colors, fabric type, pattern, fit, and other relevant fashion attributes.
                        Be specific and accurate in your analysis. If certain attributes are not clearly visible, make reasonable inferences based on what you can see.
                        Output the information in the specified JSON format."""}
                    ]
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analyze this clothing item and extract its properties."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                    ]
                )
            ]
            
            response = self.structured_llm.invoke(prompt)
            return response
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None

    def add_to_wardrobe(self, clothing_item: ClothingItem):
        """Add analyzed clothing item to user's wardrobe"""
        item_dict = json.loads(clothing_item.model_dump_json())
        item_dict['id'] = len(self.wardrobe) + 1
        self.wardrobe.append(item_dict)
        return item_dict['id']

    def get_outfit_recommendations(self, user_preferences: str, num_recommendations: int = 3):
        """Get outfit recommendations based on user preferences"""
        if not self.wardrobe:
            return None
        
        try:
            prompt = [
                SystemMessage(
                    content=[
                        {"type": "text", "text": f"""You are an expert fashion stylist. Based on the user's preferences and their wardrobe items, 
                        recommend complete outfits that match their needs. Consider color coordination, style compatibility, occasion appropriateness, and seasonal suitability.
                        
                        User's Wardrobe:
                        {json.dumps(self.wardrobe, indent=2)}
                        
                        Guidelines:
                        1. Recommend complete outfits (try to include both top and bottom wear when applicable)
                        2. Consider color harmony and style coherence
                        3. Match the occasion and season specified by the user
                        4. Provide practical styling advice
                        5. Maximum {num_recommendations} outfit recommendations
                        6. Only use item IDs that exist in the wardrobe"""}
                    ]
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"User preferences: {user_preferences}"}
                    ]
                )
            ]
            
            response = self.recommendation_llm.invoke(prompt)
            return response
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return None

    def get_item_by_id(self, item_id: str):
        """Get wardrobe item by ID"""
        for item in self.wardrobe:
            if str(item['id']) == str(item_id):
                return item
        return None

    def display_outfit_recommendation(self, recommendation: OutfitRecommendation):
        """Display outfit recommendation in a formatted way"""
        st.subheader("🎯 Recommended Outfit")
        
        # Display recommended items
        st.write("**Outfit Items:**")
        for item_id in recommendation.recommended_items:
            item = self.get_item_by_id(item_id)
            if item:
                with st.expander(f"Item {item_id}: {item['category']} - {', '.join(item['color'])}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Description:** {item['description']}")
                        st.write(f"**Fabric:** {item['fabric']}")
                        st.write(f"**Pattern:** {item['pattern']}")
                        st.write(f"**Fit:** {item['fit']}")
                    with col2:
                        st.write(f"**Occasions:** {', '.join(item['occasion'])}")
                        st.write(f"**Seasons:** {', '.join(item['season'])}")
                        st.write(f"**Features:** {', '.join(item['features'])}")
        
        # Display reasoning
        st.write("**Why this outfit works:**")
        st.write(recommendation.reasoning)
        
        # Display styling tips
        if recommendation.style_tips:
            st.write("**Styling Tips:**")
            for tip in recommendation.style_tips:
                st.write(f"• {tip}")

def get_api_key():
    """Get API key from various sources"""
    api_key = None
    
    # Try Streamlit secrets first (for deployed app)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return api_key, "secrets"
    except:
        pass
    
    # Try environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key, "environment"
    
    # Fall back to user input
    return None, "user_input"

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .stTab > div > div > div > div {
        padding-top: 2rem;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>👗 StyleSync - AI Fashion Assistant</h1>
        <p>Upload your clothing images and get personalized outfit recommendations!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'wardrobe_items' not in st.session_state:
        st.session_state.wardrobe_items = []
    if 'api_key_status' not in st.session_state:
        st.session_state.api_key_status = None
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Get API key
        api_key, source = get_api_key()
        
        if source == "secrets":
            st.success("✅ API Key loaded from secrets")
        elif source == "environment":
            st.success("✅ API Key loaded from environment")
        else:
            api_key = st.text_input(
                "Enter your Gemini API Key:", 
                type="password",
                help="Get your API key from Google AI Studio: https://makersuite.google.com/app/apikey",
                placeholder="AIza..."
            )
        
        # Initialize bot if API key is available
        if api_key and st.session_state.bot is None:
            try:
                with st.spinner("Initializing AI assistant..."):
                    st.session_state.bot = StyleSyncBot(api_key)
                    st.session_state.api_key_status = "success"
                st.success("✅ AI Assistant ready!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.session_state.api_key_status = "error"
        
        # Wardrobe statistics
        st.header("📊 Wardrobe Stats")
        if st.session_state.bot and st.session_state.bot.wardrobe:
            wardrobe_df = pd.DataFrame(st.session_state.bot.wardrobe)
            st.metric("Total Items", len(st.session_state.bot.wardrobe))
            
            # Category distribution
            if not wardrobe_df.empty:
                category_counts = wardrobe_df['category'].value_counts()
                st.write("**Categories:**")
                for category, count in category_counts.items():
                    st.write(f"• {category}: {count}")
        else:
            st.info("No items in wardrobe yet")
        
        # App info
        st.markdown("---")
        st.markdown("**About StyleSync**")
        st.caption("AI-powered fashion assistant that analyzes your clothing and creates personalized outfit recommendations.")
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue.")
        st.info("💡 Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        
        # Instructions
        st.markdown("### How to get your API key:")
        st.markdown("""
        1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy the key and paste it in the sidebar
        """)
        return
    
    if st.session_state.api_key_status == "error":
        st.error("❌ There was an issue with your API key. Please check and try again.")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📷 Add Clothing", "👔 Get Recommendations", "👗 My Wardrobe"])
    
    with tab1:
        st.header("📷 Add Clothing to Your Wardrobe")
        st.markdown("Upload clear images of individual clothing items for analysis.")
        
        uploaded_files = st.file_uploader(
            "Choose clothing images", 
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload clear, well-lit images of individual clothing items"
        )
        
        if uploaded_files and st.session_state.bot:
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📸 Analyzing: {uploaded_file.name}", expanded=idx==0):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Display image
                        try:
                            image = Image.open(uploaded_file)
                            # Resize image if too large
                            if image.size[0] > 800 or image.size[1] > 800:
                                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                            st.image(image, caption=uploaded_file.name, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
                            continue
                    
                    with col2:
                        if st.button(f"🔍 Analyze {uploaded_file.name}", key=f"analyze_{idx}_{uploaded_file.name}"):
                            with st.spinner("🤖 AI is analyzing your clothing item..."):
                                try:
                                    # Get image bytes
                                    uploaded_file.seek(0)  # Reset file pointer
                                    image_bytes = uploaded_file.getvalue()
                                    
                                    # Analyze the image
                                    clothing_item = st.session_state.bot.analyze_clothing_image(image_bytes)
                                    
                                    if clothing_item:
                                        # Add to wardrobe
                                        item_id = st.session_state.bot.add_to_wardrobe(clothing_item)
                                        
                                        st.success(f"✅ Added to wardrobe as Item #{item_id}")
                                        
                                        # Display analysis results
                                        st.json(json.loads(clothing_item.model_dump_json()))
                                        
                                        # Rerun to update sidebar stats
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to analyze the image. Please try again.")
                                except Exception as e:
                                    st.error(f"❌ Error processing image: {str(e)}")
    
    with tab2:
        st.header("👔 Get Personalized Recommendations")
        
        if not st.session_state.bot or not st.session_state.bot.wardrobe:
            st.warning("⚠️ Please add some clothing items to your wardrobe first!")
            st.info("Go to the 'Add Clothing' tab to upload and analyze your clothing images.")
        else:
            st.markdown("Tell us about the occasion and your preferences:")
            
            # User preferences input
            col1, col2 = st.columns(2)
            
            with col1:
                occasion = st.selectbox("🎯 Occasion", [
                    "Casual", "Work/Professional", "Party", "Date Night", 
                    "Beach/Pool", "Gym/Athletic", "Formal Event", "Travel"
                ])
                
                season = st.selectbox("🌤️ Season", [
                    "Spring", "Summer", "Fall", "Winter", "Any"
                ])
            
            with col2:
                time_of_day = st.selectbox("🕐 Time of Day", [
                    "Morning", "Afternoon", "Evening", "Night", "Any"
                ])
                
                style_preference = st.selectbox("✨ Style Preference", [
                    "Comfortable", "Stylish", "Professional", "Trendy", 
                    "Classic", "Minimalist", "Bold"
                ])
            
            additional_notes = st.text_area(
                "📝 Additional preferences:",
                placeholder="e.g., prefer bright colors, need pockets, avoid tight fits, specific color combinations...",
                height=100
            )
            
            # Combine preferences
            user_preferences = f"""
            Occasion: {occasion}
            Season: {season}
            Time of Day: {time_of_day}
            Style Preference: {style_preference}
            Additional Notes: {additional_notes}
            """
            
            if st.button("🎯 Get My Perfect Outfit", type="primary", use_container_width=True):
                with st.spinner("🤖 Creating your perfect outfit recommendation..."):
                    try:
                        recommendations = st.session_state.bot.get_outfit_recommendations(user_preferences)
                        
                        if recommendations:
                            st.session_state.bot.display_outfit_recommendation(recommendations)
                        else:
                            st.error("❌ Could not generate recommendations. Please try again or add more items to your wardrobe.")
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
    
    with tab3:
        st.header("👗 My Wardrobe Collection")
        
        if st.session_state.bot and st.session_state.bot.wardrobe:
            st.success(f"You have {len(st.session_state.bot.wardrobe)} items in your wardrobe!")
            
            # Display wardrobe items
            for item in st.session_state.bot.wardrobe:
                with st.expander(f"👕 Item #{item['id']}: {item['category']} - {', '.join(item['color'])}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Category:** {item['category']}")
                        st.write(f"**Colors:** {', '.join(item['color'])}")
                        st.write(f"**Pattern:** {item['pattern']}")
                        st.write(f"**Fabric:** {item['fabric']}")
                    
                    with col2:
                        st.write(f"**Fit:** {item['fit']}")
                        st.write(f"**Sleeve Length:** {item['sleeve_length']}")
                        st.write(f"**Neck Type:** {item['neck_type']}")
                        st.write(f"**Gender:** {item['gender']}")
                    
                    with col3:
                        st.write(f"**Occasions:** {', '.join(item['occasion'])}")
                        st.write(f"**Seasons:** {', '.join(item['season'])}")
                        st.write(f"**Features:** {', '.join(item['features'])}")
                    
                    st.write(f"**Description:** {item['description']}")
            
            # Export functionality
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export Wardrobe as JSON", use_container_width=True):
                    wardrobe_json = json.dumps(st.session_state.bot.wardrobe, indent=2)
                    st.download_button(
                        label="📄 Download JSON File",
                        data=wardrobe_json,
                        file_name="my_wardrobe.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("🗑️ Clear Wardrobe", use_container_width=True):
                    if st.session_state.get('confirm_clear', False):
                        st.session_state.bot.wardrobe = []
                        st.session_state.confirm_clear = False
                        st.success("✅ Wardrobe cleared!")
                        st.rerun()
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("Click again to confirm clearing your wardrobe!")
        else:
            st.info("🛍️ Your wardrobe is empty. Start by adding some clothing items!")
            st.markdown("**Tips for best results:**")
            st.markdown("""
            - Upload clear, well-lit photos
            - Show the full clothing item
            - Use a plain background when possible
            - Take photos from the front view
            """)

if __name__ == "__main__":
    main()
