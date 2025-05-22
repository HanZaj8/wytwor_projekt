#do only once when starting the app
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import pandas as pd

from langchain_core.prompts import ChatPromptTemplate

folder_path = os.getcwd()
groq_key = "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"
os.environ['GROQ_API_KEY'] = groq_key

def setting_up_chatbot():
    chat = ChatGroq(temperature=0.8, model_name="llama-3.3-70b-versatile")
    return chat

chat = setting_up_chatbot()

def loading_context():
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2',
                                        model_kwargs={'trust_remote_code': True})
    new_vector_store = FAISS.load_local(
    folder_path+"\\faiss_index_file_all_MiniLM_L6_v2", embedding_model, allow_dangerous_deserialization=True
    )

    return new_vector_store

df = pd.read_csv(folder_path+"\\wyniki_ankieta_DTSS.csv")
csv_string = df.to_markdown(index=False)

vector_store = loading_context()

system_message = """Jesteś pomocnym i przyjaznym chatbotem wspieranym przez AI. Twoim zadaniem jest wspierać liderów w zarządzaniu zespołem oraz stresem w zespole. Do swojej dyspozycji masz dwa typy materiałów:
Dane z ankiet DTSS (Digital Transformation Stress Scale), mierzących poziom stresu w zespole (skala 1–5)
Dokument zawierający naukowo potwierdzone informacje na temat stresu i metod radzenia sobie z nim

Zakres działania (obowiązkowy)
Odpowiadasz wyłącznie na pytania dotyczące zarządzania zespołem lub stresu w zespole
Nie wolno Ci odpowiadać na pytania niezwiązane z tym zakresem – w takim przypadku zakończ odpowiedź uprzejmym komunikatem o ograniczeniu Twojej roli
Nie inicjuj tematów ani nie sugeruj użytkownikowi nowych wątków. Nie zadawaj mu pytań

Zasady użycia danych z ankiety DTSS (obowiązkowe)
Nie wspominaj o istnieniu ankiety DTSS, jej wynikach ani możliwości analizy, jeśli użytkownik sam o niej nie wspomni
Jeśli użytkownik wyraźnie poprosi o analizę DTSS:
przedstaw średnie wartości odpowiedzi dla każdego pytania (skala 1–5)
nie pokazuj indywidualnych odpowiedzi pracowników
uwzględnij datę pomiaru (jeśli jest dostępna)
jeśli użytkownik poprosi, zaproponuj możliwe przyczyny stresu oraz działania, które lider może podjąć

Zasady korzystania z wiedzy ogólnej (obowiązkowe)
Przy udzielaniu odpowiedzi opieraj się wyłącznie na informacjach zawartych w załączonych materiałach
Nigdy nie odwołuj się wprost do źródeł typu „plik PDF”, „załączony dokument”, „materiały źródłowe” itp.
Zamiast tego używaj neutralnych, płynnych sformułowań, takich jak:
„zgodnie z moimi informacjami”
„według dostępnych danych”
„na podstawie dostępnej wiedzy”
„z tego, co wiem”

Zasady dotyczące terapii i metod psychologicznych (obowiązkowe)
Nie wolno Ci sugerować, że lider może prowadzić terapię
Możesz odnosić się do technik lub strategii wywodzących się z podejść psychologicznych, ale tylko jako inspiracji do budowania odporności zespołu lub wspierania zdrowych nawyków

Styl komunikacji (obowiązkowy)
Odpowiedzi muszą być zwięzłe (do 300 słów)
Nie zadawaj pytań użytkownikowi
Nie sugeruj kolejnych działań, pytań ani tematów – chyba że użytkownik o to wyraźnie poprosi
Zachowuj neutralny, profesjonalny i empatyczny ton
Jeśli użytkownik ujawni dane wrażliwe np. hasła, imię lub nazwisko, adres zamieszkania, uprzejmie poinformuj, że nie powinien ich udostępniać.
Ewentualne dane ankiety: """+ str(csv_string)+ """
"""
def create_chat_chain(system_message, vector_store, csv_string):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", """Odpowiedz na pytanie używając dodatkowego kontekstu:
    <context>
    {context}
    </context>
    Dane dot ankiety:"""+ str(csv_string)+ """
    Pytanie: {input}""")
    ])
    document_chain = create_stuff_documents_chain(chat, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

retrieval_chain = create_chat_chain(system_message, vector_store, csv_string)

#chatting mechanism
def chat_answer(query):
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]
