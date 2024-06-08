import gradio as gr


def greet(name, intensity):
    # This is where your fun code goes
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=[gr.Textbox(label="Greeting", lines=3)],
)

demo.launch()
