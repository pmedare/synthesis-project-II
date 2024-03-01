import gradio as gr

def test(name):
    return f"Hello {name}!"

app = gr.Interface(fn=test, inputs="text", outputs="text")
app.launch()