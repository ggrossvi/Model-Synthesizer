import asyncio
import datetime
import time
from pathlib import Path
import streamlit as st
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai

# 1. Initialize API Clients (Pulls automatically from your secrets/env setup)
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
REQUEST_TIMEOUT_SECONDS = 30.0
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports"

# 2. Helper Functions: Token Counting & Costs
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.33) if text else 0

def calculate_cost(model_name: str, input_text: str, output_text: str) -> float:
    in_tokens = estimate_tokens(input_text)
    out_tokens = estimate_tokens(output_text)
    rates = {
        "gpt-4o": {"in": 2.50, "out": 10.00},
        "claude-3-5-sonnet": {"in": 3.00, "out": 15.00},
        "gemini-3.6-flash": {"in": 0.30, "out": 2.50},
        "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
        "gemini-2.0-flash": {"in": 0.10, "out": 0.40}
    }
    if model_name not in rates:
        return 0.0
    return ((in_tokens / 1_000_000) * rates[model_name]["in"]) + ((out_tokens / 1_000_000) * rates[model_name]["out"])

# 3. Async API Fetch Routines with Graceful Failovers & Timeouts
async def get_openai_response(prompt, model, temp):
    openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else AsyncOpenAI()
    started_at = time.perf_counter()
    try:
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    openai_client.chat.completions.create(
                        model=model, temperature=temp, messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS
                )
                ans = response.choices[0].message.content or ""
                elapsed = time.perf_counter() - started_at
                return ans, calculate_cost("gpt-4o", prompt, ans), elapsed
            except asyncio.TimeoutError:
                if attempt < 1:
                    continue
                elapsed = time.perf_counter() - started_at
                return f"⚠️ Model Offline (Error: TimeoutError: exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s)", 0.0, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        return f"⚠️ Model Offline (Error: {type(e).__name__}: {str(e)})", 0.0, elapsed
    finally:
        close_fn = getattr(openai_client, "close", None)
        if callable(close_fn):
            close_result = close_fn()
            if asyncio.iscoroutine(close_result):
                await close_result

async def get_anthropic_response(prompt, model, temp):
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else AsyncAnthropic()
    started_at = time.perf_counter()
    try:
        request_payload = {
            "model": model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temp is not None:
            request_payload["extra_body"] = {"temperature": float(temp)}

        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    anthropic_client.messages.create(**request_payload),
                    timeout=REQUEST_TIMEOUT_SECONDS
                )
                ans = "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")
                elapsed = time.perf_counter() - started_at
                return ans, calculate_cost("claude-3-5-sonnet", prompt, ans), elapsed
            except asyncio.TimeoutError:
                if attempt < 1:
                    continue
                elapsed = time.perf_counter() - started_at
                return f"⚠️ Model Offline (Error: TimeoutError: exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s)", 0.0, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        return f"⚠️ Model Offline (Error: {type(e).__name__}: {str(e)})", 0.0, elapsed
    finally:
        close_fn = getattr(anthropic_client, "close", None)
        if callable(close_fn):
            close_result = close_fn()
            if asyncio.iscoroutine(close_result):
                await close_result

async def get_gemini_response(prompt, model, temp):
    started_at = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
        gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else genai.Client()

        candidate_models = [model, "gemini-3.6-flash", "gemini-2.5-flash"]
        last_error = None

        for candidate_model in dict.fromkeys(candidate_models):
            def send_message():
                chat = gemini_client.chats.create(model=candidate_model)
                return chat.send_message(
                    message=prompt,
                    config={"temperature": temp}
                )

            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        loop.run_in_executor(None, send_message),
                        timeout=REQUEST_TIMEOUT_SECONDS
                    )
                    ans = response.text or ""
                    elapsed = time.perf_counter() - started_at
                    return ans, calculate_cost(candidate_model, prompt, ans), elapsed
                except Exception as e:
                    last_error = e
                    message = str(e)

                    # If the model name is retired/unavailable, try the next fallback model.
                    if "NOT_FOUND" in message:
                        break

                    # Retry transient capacity/rate failures before giving up.
                    transient = isinstance(e, asyncio.TimeoutError) or any(token in message for token in ["UNAVAILABLE", "503", "RESOURCE_EXHAUSTED", "429"])
                    if transient and attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue

                    # For non-transient errors, fail immediately.
                    raise

        raise last_error if last_error else RuntimeError("No Gemini model candidates available")
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started_at
        return f"⚠️ Model Offline (Error: TimeoutError: exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s)", 0.0, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        return f"⚠️ Model Offline (Error: {type(e).__name__}: {str(e)})", 0.0, elapsed

async def get_all_responses(prompt, temp, use_openai, use_anthropic, use_gemini):
    responses = {
        "openai": ("⏭️ Skipped (disabled in sidebar)", 0.0),
        "anthropic": ("⏭️ Skipped (disabled in sidebar)", 0.0),
        "gemini": ("⏭️ Skipped (disabled in sidebar)", 0.0),
    }

    task_map = {}
    if use_openai:
        task_map["openai"] = get_openai_response(prompt, "gpt-4o", temp)
    if use_anthropic:
        task_map["anthropic"] = get_anthropic_response(prompt, "claude-3-5-sonnet-latest", temp)
    if use_gemini:
        task_map["gemini"] = get_gemini_response(prompt, "gemini-3.6-flash", temp)

    if task_map:
        results = await asyncio.gather(*task_map.values())
        for key, value in zip(task_map.keys(), results):
            responses[key] = value

    return responses


def model_success(ans_text: str) -> bool:
    return not ans_text.startswith("⚠️ Model Offline") and not ans_text.startswith("⏭️ Skipped")


def save_markdown_export(file_name: str, content: str) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORTS_DIR / file_name
    export_path.write_text(content, encoding="utf-8")
    return export_path


def build_history_markdown(history_items) -> str:
    md_content = "# 🤖 AI Aggregator History Export\n\n"
    for i, item in enumerate(history_items, 1):
        md_content += f"## Query {i}: {item['prompt']}\n\n"
        if item['synthesis']:
            md_content += f"### ✨ Synthesized Consensus Answer\n{item['synthesis']}\n\n"
        md_content += f"### 🔍 Raw Source Models used:\n"
        md_content += f"* **GPT-4o:** {item['raw_openai']}\n"
        md_content += f"  * Time: {item.get('raw_openai_time', 0.0):.2f}s\n\n"
        md_content += f"* **Claude 3.5:** {item['raw_anthropic']}\n"
        md_content += f"  * Time: {item.get('raw_anthropic_time', 0.0):.2f}s\n\n"
        md_content += f"* **Gemini 3.6 Flash:** {item['raw_gemini']}\n"
        md_content += f"  * Time: {item.get('raw_gemini_time', 0.0):.2f}s\n\n"
        if item.get('synthesis_time') is not None:
            md_content += f"### ⏱️ Synthesis Time\n{item.get('synthesis_time', 0.0):.2f}s\n\n"
        md_content += f"---\n\n"
    return md_content

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
    st.markdown("### 🧠 Models to Query")
    use_openai = st.checkbox("GPT-4o", value=True)
    use_anthropic = st.checkbox("Claude 3.5 Sonnet", value=True)
    use_gemini = st.checkbox("Gemini (gemini-3.6-flash)", value=True)
    
    st.divider()
    st.markdown("### 📊 Session Accumulator")
    st.metric(label="Total Running Cost (USD)", value=f"${st.session_state.total_spend:.5f}")
    if st.button("Reset Counter"):
        st.session_state.total_spend = 0.0
        st.session_state.history = []
        st.rerun()
        
# --- MAIN UI WORKSPACE ---
st.title("🤖 Custom AI Aggregator & Synthesizer")
user_input = st.text_area("Enter your question:", placeholder="Type your query here...", height=120)

run_synthesis = st.checkbox("⚡ Synthesize final master answer", value=True, help="Uncheck this to skip synthesis fees and read raw comparisons only.")

if st.button("Fetch Responses", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a prompt first.")
    elif not any([use_openai, use_anthropic, use_gemini]):
        st.warning("Select at least one model in the sidebar.")
    else:
        query_total = 0.0
        
        # Step 1: Run the baseline models with robust error safety blocks
        with st.spinner("Fetching raw insights from models..."):
            responses = asyncio.run(
                get_all_responses(
                    user_input,
                    temperature,
                    use_openai=use_openai,
                    use_anthropic=use_anthropic,
                    use_gemini=use_gemini,
                )
            )
            (ans_o, cost_o, time_o) = responses["openai"]
            (ans_a, cost_a, time_a) = responses["anthropic"]
            (ans_g, cost_g, time_g) = responses["gemini"]
        query_total += (cost_o + cost_a + cost_g)
        
        # Step 2: Extract which models actually completed successfully for dynamic synthesis filtering
        available_insights = []
        if use_openai and model_success(ans_o):
            available_insights.append(f"GPT-4o:\n{ans_o}")
        if use_anthropic and model_success(ans_a):
            available_insights.append(f"Claude 3.5:\n{ans_a}")
        if use_gemini and model_success(ans_g):
            available_insights.append(f"Gemini:\n{ans_g}")
        
        best_answer = ""
        cost_synth = 0.0
        synthesis_time = None
        
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
                    synth_started_at = time.perf_counter()
                    # Pick the first available enabled provider for synthesis.
                    if use_anthropic and model_success(ans_a):
                        best_answer, cost_synth = asyncio.run(get_anthropic_response(synthesis_prompt, "claude-3-5-sonnet-latest", 0.2))
                    elif use_openai and model_success(ans_o):
                        best_answer, cost_synth = asyncio.run(get_openai_response(synthesis_prompt, "gpt-4o", 0.2))
                    elif use_gemini and model_success(ans_g):
                        best_answer, cost_synth = asyncio.run(get_gemini_response(synthesis_prompt, "gemini-3.6-flash", 0.2))
                    else:
                        best_answer = "❌ Synthesis unavailable: no enabled model returned a successful response."
                        st.error(best_answer)
                    synthesis_time = time.perf_counter() - synth_started_at
                        
                if cost_synth > 0 or best_answer:
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
                tab1, tab2, tab3 = st.tabs(["GPT-4o", "Claude 3.5 Sonnet", "Gemini 3.6 Flash"])
                with tab1: st.write(ans_o)
                with tab2: st.write(ans_a)
                with tab3: st.write(ans_g)
                    
        with col2:
            with st.expander("📈 Query Cost Breakdown", expanded=True):
                st.write(f"**GPT-4o:** ${cost_o:.5f}")
                st.caption(f"Time: {time_o:.2f}s")
                st.write(f"**Claude 3.5:** ${cost_a:.5f}")
                st.caption(f"Time: {time_a:.2f}s")
                st.write(f"**Gemini 3.6 Flash:** ${cost_g:.5f}")
                st.caption(f"Time: {time_g:.2f}s")
                if run_synthesis:
                    st.write(f"**Synthesis Layer:** ${cost_synth:.5f}")
                    if synthesis_time is not None:
                        st.caption(f"Time: {synthesis_time:.2f}s")
                st.markdown(f"**Total for this prompt:** `${query_total:.5f}`")
        
        # Step 5: Update tracking session arrays
        st.session_state.total_spend += query_total
        st.session_state.history.append({
            "prompt": user_input,
            "synthesis": best_answer if run_synthesis else None,
            "synthesis_time": synthesis_time,
            "raw_openai": ans_o,
            "raw_openai_time": time_o,
            "raw_anthropic": ans_a,
            "raw_anthropic_time": time_a,
            "raw_gemini": ans_g
            ,"raw_gemini_time": time_g
        })

# --- EXPORT WORKSPACE ---
with st.sidebar:
    st.divider()
    st.markdown("### 💾 Export Workspace")
    if st.session_state.history:
        md_content = build_history_markdown(st.session_state.history)

        last_prompt = st.session_state.history[-1]['prompt']
        short_topic = "_".join(last_prompt.split()[:4]).lower()
        short_topic = "".join(c for c in short_topic if c.isalnum() or c == "_")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_filename = f"{short_topic}_{timestamp}.md"

        st.download_button(
            label="Download Session Logs (.md)",
            data=md_content,
            file_name=dynamic_filename,
            mime="text/markdown",
            use_container_width=True
        )
        if st.button("Save Session Logs to exports/", use_container_width=True):
            saved_path = save_markdown_export(dynamic_filename, md_content)
            st.success(f"Saved local copy to {saved_path}")
    else:
        st.info("Run a query to generate history logs.")
