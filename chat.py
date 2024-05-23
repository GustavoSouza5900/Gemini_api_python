import google.generativeai as genai
from pathlib import Path # usando apenas pra extrair minha api_key
    
#add your google api key, you can get it here: https://aistudio.google.com/app/prompts/new_chat
API_KEY = Path('/home/asuka5900/API_KEY').read_text().rstrip()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

chat = model.start_chat()
while True:
    prompt = input('Digite o prompt: ')
    # O comando fim encerra o loop
    if prompt == 'fim':
        break
    # O comando reset reinicia o chat apagando o histórico
    elif prompt == 'reset':
        chat = model.start_chat()
        continue

    response = chat.send_message(prompt)
    print(f'\n{response.text}\n')

