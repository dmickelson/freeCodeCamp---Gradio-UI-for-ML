# Gradio Python client makes it very easy to use any Gradio app as an API.

# The Gradio client works with any hosted Gradio app!

# using callback
from gradio_client import Client, file

# client = Client("abidlabs/whisper")
# client.predict(
#     audio=file("audio_sample.wav")
# )


def print_result(x):
    print("The translated result is: {x}")


client = Client("abidlabs/en2fr")
client.view_api()

# Blocking call to endpoint
# print(f'Translation: {client.predict("Hello")}')

# non-blocking with callback
job = client.submit("Hello", api_name="/predict",
                    result_callbacks=[print_result])

# Potential mutliple outputs if not using callback
print(job.result())
print(job.outputs())

# Job status
print(job.status())
