import streamlit as st
from rag-model import top_agent, base_query_engine  # Import your agents and engine setup from your script

# Example function to perform a query using the top-level agent
def query_top_agent(question):
    response = top_agent.query(question)
    return response

# Example function to perform a query using the baseline query engine
def query_baseline_engine(question):
    response = base_query_engine.query(question)
    return str(response)  # Convert the response to string if necessary

# Streamlit app layout
def main():
    st.title("Multi-Document Agent Query System")

    st.write("## Ask a question")
    question = st.text_input("Enter your question here:")

    if question:  # Check if a question has been entered
        st.write("### Top Agent Response")
        top_agent_response = query_top_agent(question)
        st.write(top_agent_response)
        
        st.write("### Baseline Engine Response")
        baseline_response = query_baseline_engine(question)
        st.write(baseline_response)

if __name__ == "__main__":
    main()
