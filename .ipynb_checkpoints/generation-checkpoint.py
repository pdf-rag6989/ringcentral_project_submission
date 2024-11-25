from openai import OpenAI
import time
from pydantic import BaseModel, Field
from typing import Optional
from generic import GenericFunction
from logger import Logger
import wandb
import os
import time

class CustomExtraction(BaseModel):
    car_model_name: Optional[str] = Field(None, description="The name of the car model if matched.")
    image_url: Optional[str] = Field(None, description="The URL of the image if provided.")
    text_or_image_search: str = Field(..., description="Indicates whether the input is text or an image search.")
    
class Results_Generation:
    def __init__(self):
        self.logger = Logger()
        self.generic = GenericFunction()
        self.OPENAI_API_KEY = self.generic.get_value("api_keys")['openai']
        self.MAX_RETRIES = 3
        self.SLEEP_DURATION = 5
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.prompt= self.read_prompt("./prompts/entity_extraction.txt")
        self.final_prompt=self.read_prompt("./prompts/final_generation.txt")
        if not wandb.run:
            wandb.login(key="4416a87872cc6338f305ea2d6751b22b8e5e2fc0")
            wandb.init(project="PDF RAG on Car Manuals", name="Test Run", config={
            "model": "gpt-4-turbo",
            "max_retries": self.MAX_RETRIES,
            "temperature": 0.7,
            "max_tokens": 800
                })
        self.table = wandb.Table(columns=["Input", "Response","Status","Elapsed Time"])

    def read_prompt(self,file_path):
        with open(file_path, 'r') as file:
            return file.read().strip()
            
    def base_gpt4_model(self, conversation_history):
        """
        Generate a response using GPT-4 Turbo.
        """
        start_time = time.time()
        for _ in range(self.MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                elapsed_time = time.time() - start_time
                response=completion.choices[0].message.content.strip()
                self.table.add_data(conversation_history, response,'Passed', elapsed_time)
                wandb.log({"Dynamic Table Results": self.table})
                return response
            except Exception as e:
                if _ == self.MAX_RETRIES - 1:
                    self.table.add_data(conversation_history, response,'Failed', elapsed_time)
                    wandb.log({"Dynamic Table Results": self.table})
                    return "An error occurred: {}".format(e)
                time.sleep(self.SLEEP_DURATION)

    def extract_entities(self, user_input):
        """
        Extract custom entities including car_model_name, image_url, and text_or_image_search.
        """
        pdf_list=self.generic.get_value('file_mappings')
        pdf_file_names = [list(item.values())[0] for item in pdf_list]
        self.logger.log_info (pdf_file_names)
        prompt_updated=self.prompt.format(pdf_file_names=", ".join(pdf_file_names),user_input=user_input)
        for _ in range(self.MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "Extract the following fields in JSON: car_model_name, image_url (if present) and text_or_image_search"},
                        {"role": "user", "content": prompt_updated},
                    ],
                )             
                response_text =  completion.choices[0].message.content
                response_json = CustomExtraction.parse_raw(response_text)
                return response_json.dict()
                #return response_text
            except Exception as e:
                if _ == self.MAX_RETRIES - 1:
                    return {"error": str(e)}
                time.sleep(self.SLEEP_DURATION)

