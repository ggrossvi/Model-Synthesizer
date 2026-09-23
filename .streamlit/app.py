import asyncio
import datetime
import streamlit as st
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai

# 1. Initialize API Clients (Pulls automatically from your secrets/env setup)
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)

openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else AsyncOpenAI()
anthropic_client = AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else AsyncAnthropic()

# 2. Helper Functions: Token Counting & Costs
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.33) if text else 0

def calculate_cost(model_name: str, input_text: str, output_text: str) -> float:
    in_tokens = estimate_tokens(input_text)
    out_tokens = estimate_tokens(output_text)
    rates = {
        "gpt-4o": {"in": 2.50, "out": 10.00},
        "claude-3-5-sonnet": {"in": 3.00, "out": 15.00},
        "gemini-1.5-pro": {"in": 1.25, "out": 5.00}
    }
    if model_name not in rates:
        return 0.0
    return ((in_tokens / 1_000_000) * rates[model_name]["in"]) + ((out_tokens / 1_000_000) * rates[model_name]["out"])

# 3. Async API Fetch Routines with Graceful Failovers & Timeouts
async def get_openai_response(prompt, model, temp):
    try:
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=model, temperature=temp, messages=[{"role": "user", "content": prompt}]
            ),
            timeout=15.0
        )
        ans = response.choices[0].message.content or ""
        return ans, calculate_cost("gpt-4o", prompt, ans)
    except Exception as e:
        return f"⚠️ Model Offline (Error: {type(e).__name__}: {str(e)})", 0.0

async def get_anthropic_response(prompt, model, temp):
    try:
        response = await asyncio.wait_for(
            anthropic_client.messages.create(
                model=model, max_tokens=2048, temperature=temp, messages=[{"role": "user", "content": prompt}]
            ),
            timeout=15.0
        )
        ans = "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return ans, calculate_cost("claude-3-5-sonnet", prompt, ans)
    except Exception as e:
        return f"⚠️ Model Offline (Error: {type(e).__name__}: {str(e)})", 0.0

async def get_gemini_response(prompt, model, temp):
    try:
        loop = asyncio.get_running_loop()
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
        gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else genai.Client()

        def send_message():
            chat = gemini_client.chats.create(model=model)
            return chat.send_message(
                message=prompt,
                config={"temperature": temp}
            )

        response = await asyncio.wait_for(
            loop.run_in_executor(None, send_message),
            timeout=15.0
        )
        ans = response.text or ""
        return ans, calculate_cost("gemini-1.5-pro", prompt, ans)
    except Exception as e:
        return f"⚠️ Model Offline (Error: {type(e).__name__}: {str(e)})", 0.0

async def get_all_responses(prompt, temp):
    tasks = [
        get_openai_response(prompt, "gpt-4o", temp),
        get_anthropic_response(prompt, "claude-3-5-sonnet-latest", temp),
        get_gemini_response(prompt, "gemini-1.5-pro", temp)
    ]
    return await asyncio.gather(*tasks)

# 4. App Workspace Setup
st.set_page_config(page_title="AI Aggregator Pro", layout="wide")

# Persistent State Management for Cost Tracking and History Log
if "total_spend" not in st.session_state:
    st.session_state.total_spend = 0.0
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR Configuration ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3, 0.1)
    
    st.divider()
    st.markdown("### 📊 Session Accumulator")
    st.metric(label="Total Running Cost (USD)", value=f"${st.session_state.total_spend:.5f}")
    if st.button("Reset Counter"):
        st.session_state.total_spend = 0.0
        st.session_state.history = []
        st.rerun()
        
    st.divider()
    st.markdown("### 💾 Export Workspace")
    if st.session_state.history:
        # Build Markdown file text content strings on the fly
        md_content = "# 🤖 AI Aggregator History Export\n\n"
        for i, item in enumerate(st.session_state.history, 1):
            md_content += f"## Query {i}: {item['prompt']}\n\n"
            if item['synthesis']:
                md_content += f"### ✨ Synthesized Consensus Answer\n{item['synthesis']}\n\n"
            md_content += f"### 🔍 Raw Source Models used:\n"
            md_content += f"* **GPT-4o:** {item['raw_openai']}\n\n"
            md_content += f"* **Claude 3.5:** {item['raw_anthropic']}\n\n"
            md_content += f"* **Gemini 1.5 Pro:** {item['raw_gemini']}\n\n"
            md_content += f"---\n\n"
            
        # Clean the last prompt into a shortened snake_case file prefix
        last_prompt = st.session_state.history[-1]['prompt']
        short_topic = "_".join(last_prompt.split()[:4]).lower()
        short_topic = "".join(c for c in short_topic if c.isalnum() or c == "_")
        
        # Build Timestamp layout string: YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_filename = f"{short_topic}_{timestamp}.md"
            
        st.download_button(
            label="Download Session Logs (.md)",
            data=md_content,
            file_name=dynamic_filename,
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("Run a query to generate history logs.")

# --- MAIN UI WORKSPACE ---
st.title("🤖 Custom AI Aggregator & Synthesizer")
user_input = st.text_area("Enter your question:", placeholder="Type your query here...", height=120)

run_synthesis = st.checkbox("⚡ Synthesize final master answer", value=True, help="Uncheck this to skip synthesis fees and read raw comparisons only.")

if st.button("Fetch Responses", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        query_total = 0.0
        
        # Step 1: Run the baseline models with robust error safety blocks
        with st.spinner("Fetching raw insights from models..."):
            (ans_o, cost_o), (ans_a, cost_a), (ans_g, cost_g) = asyncio.run(get_all_responses(user_input, temperature))
        query_total += (cost_o + cost_a + cost_g)
        
        # Step 2: Extract which models actually completed successfully for dynamic synthesis filtering
        available_insights = []
        if not ans_o.startswith("⚠️ Model Offline"): available_insights.append(f"GPT-4o:\n{ans_o}")
        if not ans_a.startswith("⚠️ Model Offline"): available_insights.append(f"Claude 3.5:\n{ans_a}")
        if not ans_g.startswith("⚠️ Model Offline"): available_insights.append(f"Gemini 1.5 Pro:\n{ans_g}")
        
        best_answer = ""
        cost_synth = 0.0
        
        # Step 3: Run dynamic synthesis layer if requested and at least one model succeeded
        if run_synthesis:
            if not available_insights:
                best_answer = "❌ Critical System Error: All upstream engine requests failed. Unable to synthesize a response."
                st.error(best_answer)
            else:
                formatted_insights = "\n\n---\n\n".join(available_insights)
                synthesis_prompt = f"""
                You are an expert consensus engine. Review the available working engine models below.
                Extract the most accurate facts, discard contradictions or hallucinations, and merge the unique insights 
                into one masterfully structured, comprehensive best answer.

                Original Prompt: {user_input}

                === TARGET MATERIAL SOURCE MATERIAL ===
                {formatted_insights}
                """
                with st.spinner("Synthesizing responsive models into best answer..."):
                    # Use OpenAI as synthesis fallback if Claude fails
                    if not ans_a.startswith("⚠️ Model Offline"):
                        best_answer, cost_synth = asyncio.run(get_anthropic_response(synthesis_prompt, "claude-3-5-sonnet-latest", 0.2))
                    else:
                        best_answer, cost_synth = asyncio.run(get_openai_response(synthesis_prompt, "gpt-4o", 0.2))
                        
                query_total += cost_synth
                st.success("✨ Synthesized Best Answer")
                st.write(best_answer)
                st.divider()
        else:
            st.info("💡 Side-by-side mode activated. Check the tabs below to read individual responses.")

        # Step 4: UI Layout Breakdown
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("🔍 View Raw Underlying Source Answers", expanded=not run_synthesis):
                tab1, tab2, tab3 = st.tabs(["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"])
                with tab1: st.write(ans_o)
                with tab2: st.write(ans_a)
                with tab3: st.write(ans_g)
                    
        with col2:
            with st.expander("📈 Query Cost Breakdown", expanded=True):
                st.write(f"**GPT-4o:** ${cost_o:.5f}")
                st.write(f"**Claude 3.5:** ${cost_a:.5f}")
                st.write(f"**Gemini 1.5 Pro:** ${cost_g:.5f}")
                if run_synthesis:
                    st.write(f"**Synthesis Layer:** ${cost_synth:.5f}")
                st.markdown(f"**Total for this prompt:** `${query_total:.5f}`")
        
        # Step 5: Update tracking session arrays
        st.session_state.total_spend += query_total
        st.session_state.history.append({
            "prompt": user_input,
            "synthesis": best_answer if run_synthesis else None,
            "raw_openai": ans_o,
            "raw_anthropic": ans_a,
            "raw_gemini": ans_g
        })
