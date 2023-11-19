# caption generator
import transformers
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
import torch
import urllib
from urllib import request
from io import BytesIO
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer


flower_names = torch.load("flower_names.pt")

class CaptionCreator:
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu" # Set device to GPU if its available

    checkpoint = "microsoft/git-base"
    self.processor = AutoProcessor.from_pretrained(checkpoint) # We would load a tokenizer for language. Here we load a processor to process images
    self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)

  def get_caption(self, image):
    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    pixel_values = inputs.pixel_values.to(self.device)
    generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

def getImage(image_url):
    with urllib.request.urlopen(image_url) as image_url:
      image = Image.open(BytesIO(image_url.read()))
    return image

caption_creator = CaptionCreator()

# speciefinder

class SpecieFinder:
  def __init__(self, flower_names):
    self.flower_names = flower_names
    self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    self.flower_names_embeddings = torch.tensor(self.model.encode(flower_names))

  def get_closest_species(self, caption, k):
    caption_embedding = torch.tensor(self.model.encode([caption]))
    similarities = torch.matmul(self.flower_names_embeddings, caption_embedding.view(-1, 1))
    _, indices = similarities.topk(k,dim=0)
    closest_species = []
    for idx in indices.view(-1):
      closest_species.append(self.flower_names[idx.item()])
    return closest_species

def getFileName(name):
   return "100flowers/"+name+".jpeg"

def display(images):
  cols = st.columns(3)
  for i in range(len(images)):
        with cols[i]:
                st.image(getFileName(images[i]), use_column_width=True, caption=images[i])

def processRequest(image):
  caption = caption_creator.get_caption(image)
  specie = SpecieFinder(flower_names).get_closest_species(caption,1)[0]
  cols = st.columns(3)
  with cols[0]:
    st.image(image, caption=specie, use_column_width=True)
  neighbors = SpecieFinder(flower_names).get_closest_species(specie, 3)
  display(neighbors)

def main():
  st.title("Flower Identifier App")
  uploaded_file = st.file_uploader("Choose an image...", type="jpg")
  user_input = st.text_input("Provide an image link: ",value="https://assets.americanmeadows.com/media/catalog/product/i/r/iris-siberica-caesars-brother-credit-walters-gardens_1.jpg")
  submit_button = st.button("Submit")
  if submit_button:
    if user_input:
      image = getImage(user_input)
      processRequest(image)
    
    elif uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        processRequest(image)
    

main()