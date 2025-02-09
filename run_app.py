import tablemage as tm


tm.use_agents()


# tm.agents.options.set_llm(
#     llm_type="groq",
#     model_name="llama-3.3-70b-versatile",
#     temperature=0.1
# )
tm.agents.options.set_llm(llm_type="openai", model_name="gpt-4o", temperature=0.1)
tm.agents.options.set_multimodal_llm(
    llm_type="openai", model_name="gpt-4o", temperature=0.1
)

tm.agents.ChatDA_UserInterface(
    split_seed=42,
    memory_size=500,
    python_only=False,
    tools_only=True,
    tool_rag_top_k=5,
    multimodal=True,
).run(debug=False)
