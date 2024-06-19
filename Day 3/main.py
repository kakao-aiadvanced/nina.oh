# $ cd ./Day 3
# $ streamlit run main.py
from graph.graph_builder import generate_workflow
from ui.ui_generator import draw_ui_with_streamlit

workflow = generate_workflow()
app = workflow.compile()
# RAG scenario
inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished running: {key}:")
print(value["generation"])

draw_ui_with_streamlit(app)
