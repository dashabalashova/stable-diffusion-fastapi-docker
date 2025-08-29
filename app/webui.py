import gradio as gr
from .sd3 import generate_image

def infer(prompt):
    image = generate_image(prompt)
    return image

def launch_gradio():
    demo = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(label="Prompt", placeholder="A small dragon drinking coffee in Amsterdam, watercolor"),
        outputs=gr.Image(type="pil"),
        title="Stable Diffusion 3.5 Turbo WebUI",
        description="Enter a prompt and generate an image with Stable Diffusion 3.5 Turbo"
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()
