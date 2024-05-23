import numpy as np
import pandas as pd
from google import generativeai as genai
from pathlib import Path

def listar():
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(m.name)


def embed_df(model, title, content):
    return genai.embed_content(model=model, title=title, content=content, task_type='RETRIEVAL_DOCUMENT')['embedding']


def consultar(model, consulta, base):
    embed_consulta = genai.embed_content(model=model, content=consulta, task_type='RETRIEVAL_QUERY')['embedding']

    produtos_escalares = np.dot(np.stack(base["embeddings"]), embed_consulta)
    indice = np.argmax(produtos_escalares)
    
    return base.iloc[indice]["content"]


GOOGLE_API_KEY = Path('/home/asuka5900/API_KEY').read_text().rstrip()
genai.configure(api_key=GOOGLE_API_KEY)

blender = {
        'title': 'Blender',
        'content': 'Blender is a public project hosted on blender.org, licensed as GNU GPL, owned by its contributors.'
        }
ryujinx = {
        'title': 'Ryujinx',
        'content': 'Ryujinx is an open-source Nintendo Switch emulator created by gdkchan and written in C#. This emulator aims at providing excellent accuracy and performance, a user-friendly interface, and consistent builds.'
        }
hyprland = {
        'title': 'Hyprland',
        'content': "Hyprland is a dynamic tiling Wayland compositor based on wlroots that doesn't sacrifice on its looks. \nIt provides the latest Wayland features, is highly customizable, has all the eyecandy, the most powerful plugins, easy IPC, much more QoL stuff than other wlr-based compositors and more... "
        }

documents = [blender, ryujinx, hyprland]

df = pd.DataFrame(documents)

model = 'models/embedding-001'
df["embeddings"] = df.apply(lambda row: embed_df(model, row["title"], row["content"]), axis=1)

consulta = input('faça a consulta: ')
print(consultar(model, consulta, df))

