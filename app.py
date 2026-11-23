
import time
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# LOAD RAG SYSTEM
# ==========================================

@st.cache_resource
def load_rag():
    import rag
    return rag


# ==========================================
# HELPER FUNCTION
# ==========================================

def get_snippet(text, limit=250):

    text = text.strip().replace("\n", " ")

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0] + "..."


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #101827;
}

[data-testid="stSidebar"] * {
    color: white;
}

.stChatMessage {
    border-radius: 12px;
}

h1 {
    color: #155eef;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 Book Assistant")

    st.write("Local RAG-Based Search System")

    st.divider()

    st.success("Local Retrieval Mode")

    st.caption("No API | No Gemini | No API Quota")

    st.divider()

    st.caption("Project Technologies")

    st.write("• Python")
    st.write("• Sentence Transformers")
    st.write("• FAISS Vector Database")
    st.write("• Streamlit")

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# MAIN HEADER
# ==========================================

st.title("📚 Database Book Assistant")

st.write(
    "Ask questions from Modern Database Management, "
    "10th Edition."
)

st.caption(
    "Retrieval-Augmented Generation (RAG) | Local Search"
)

st.divider()


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# LOAD MODEL
# ==========================================

try:

    with st.spinner("Loading RAG system..."):

        rag = load_rag()

    st.success("RAG system ready!")

except Exception as error:

    st.error(f"Unable to load RAG system: {error}")

    st.stop()


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**{rag.get_source_label(source)}**"
                    )

                    st.caption(
                        f"Distance: {source['distance']:.4f}"
                    )

                    st.write(
                        get_snippet(source["text"], 250)
                    )

                    with st.expander("View Full Passage"):

                        st.write(source["text"])

                    st.divider()

        if message.get("timing"):

            st.caption(message["timing"])


# ==========================================
# CHAT INPUT
# ==========================================

query = st.chat_input(
    "Ask a question about databases..."
)


if query:

    # ==========================================
    # USER MESSAGE
    # ==========================================

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):

        st.markdown(query)


    # ==========================================
    # RETRIEVAL
    # ==========================================

    with st.chat_message("assistant"):

        with st.spinner("Searching your book..."):

            try:

                start_time = time.time()

                sources = rag.search_database(query)

                search_time = time.time() - start_time


                # ==========================================
                # CREATE SHORT RESPONSE
                # ==========================================

                if not sources:

                    answer = "No relevant information found in the book."

                else:

                    top_sources = sources[:3]

                    answer = "### Retrieved Information\n\n"

                    for source in top_sources:

                        answer += (
                            f"**{rag.get_source_label(source)}**\n\n"
                            f"{get_snippet(source['text'], 250)}\n\n"
                        )


                # ==========================================
                # PERFORMANCE
                # ==========================================

                total_time = time.time() - start_time

                timing = (
                    f"Search: {search_time:.2f}s | "
                    f"Total: {total_time:.2f}s"
                )


                # ==========================================
                # DISPLAY ANSWER
                # ==========================================

                st.markdown(answer)


                # ==========================================
                # SOURCES
                # ==========================================

                if sources:

                    with st.expander("View Sources"):

                        for source in sources[:3]:

                            st.markdown(
                                f"**{rag.get_source_label(source)}**"
                            )

                            st.caption(
                                f"Distance: {source['distance']:.4f}"
                            )

                            st.write(
                                get_snippet(source["text"], 250)
                            )

                            with st.expander("View Full Passage"):

                                st.write(source["text"])

                            st.divider()


                # ==========================================
                # TIMING
                # ==========================================

                st.caption(timing)


                # ==========================================
                # SAVE HISTORY
                # ==========================================

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources[:3],
                    "timing": timing
                })


            except Exception as error:

                st.error(f"Error: {error}")
