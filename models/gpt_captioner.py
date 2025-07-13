from transformers import pipeline

class CaptionGenerator:
    def __init__(self, model_name='gpt2'):
        self.generator = pipeline('text-generation', model=model_name)

    def generate_caption(self, relations):
        prompt = "Describe the scene: " + ", ".join(relations)
        outputs = self.generator(prompt, max_length=100, num_return_sequences=1)
        return outputs[0]['generated_text']
