# EchoSummary Gold

EchoSummary to aplikacja webowa oparta o **Streamlit**, która umożliwia:

- Wgrywanie plików audio i wideo (`mp3`, `wav`, `m4a`, `mp4`, `mov`, `webm`, `mkv`)
- Ekstrakcję i podgląd ścieżki audio z pliku wideo
- Wizualizację przebiegu fali dźwiękowej (waveform)
- **Transkrypcję mowy na tekst** (OpenAI API, domyślnie model `gpt-4o-mini-transcribe`)
- **Diarizację mówców** (Hugging Face `pyannote.audio`)
- Automatyczne **streszczanie transkrypcji** (OpenAI API, model `gpt-4o-mini`)
- Pobieranie wyników (TXT z transkrypcją oraz streszczeniem, MP3 z wyodrębnionym audio)

---

## 🚀 Wymagania

- Python 3.10
- Klucz API do [OpenAI](https://platform.openai.com/)
- Token do [Hugging Face](https://huggingface.co/settings/tokens)

### Biblioteki Python

- `streamlit`
- `openai`
- `huggingface_hub`
- `pyannote.audio`
- `pydub`
- `plotly`
- `numpy`

---

## ⚙️ Instalacja

1. Sklonuj repozytorium lub skopiuj pliki projektu:
   ```bash
   git clone https://github.com/twoj-user/echosummary.git
   cd echosummary
   ```

2. Utwórz i aktywuj środowisko wirtualne:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate    # Windows
   ```

3. Zainstaluj wymagane paczki:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Uruchomienie aplikacji

```bash
streamlit run app.py
```

Następnie otwórz przeglądarkę pod adresem: [http://localhost:8501](http://localhost:8501)

---

## 🔑 Konfiguracja kluczy

- **OpenAI API Key** – wymagany do transkrypcji i generowania streszczeń  
- **Hugging Face Token** – wymagany do diarizacji mówców (`pyannote.audio`)

Możesz je wprowadzić w **pasku bocznym aplikacji**.

---

## 📂 Funkcjonalności krok po kroku

1. **Wgraj plik** audio lub wideo.  
2. Jeśli plik to wideo → wyodrębnij ścieżkę audio.  
3. Podejrzyj waveform nagrania.  
4. Uruchom **transkrypcję z diarizacją**.  
5. Pobierz lub edytuj transkrypcję.  
6. Wygeneruj **streszczenie** (max 300 słów).  
7. Pobierz wyniki (TXT/MP3).  

---

## 🛠️ Dostosowanie

W pliku `app.py` można zmienić używany model transkrypcji.  
Domyślnie jest to:

```python
model="gpt-4o-mini-transcribe"
```

Możesz go podmienić np. na klasyczny **Whisper-1**:

```python
model="whisper-1"
```

---

## 📜 Licencja

Projekt dostępny na licencji MIT – możesz go dowolnie modyfikować i rozwijać.
