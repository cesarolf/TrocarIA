# Traço | Escrita

Protótipo local para transcrever escrita manuscrita em português e revisar possíveis diferenças ortográficas. O OCR usa EasyOCR; sem texto esperado, os apontamentos dependem de um dicionário e precisam de revisão humana.

## Executar

Recomendado: Python 3.11 ou 3.12.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

O primeiro uso baixa os pesos do OCR e pode levar alguns minutos. A aplicação processa a imagem localmente e não mantém um cadastro de pacientes. O arquivo JSON exportado deve ser tratado como dado sensível.

## Escopo e limites

- O EasyOCR em português é um OCR geral, não ajustado especificamente para escrita clínica manuscrita; pode errar, especialmente com foto desfocada, letra cursiva ou imagem inclinada.
- Para sugerir as dez categorias dos slides, informe o texto-alvo. Sem referência, a checagem de dicionário só sugere palavras fora do vocabulário.
- As categorias são heurísticas sobre a transcrição, não diagnósticos. "Letras parecidas" e "inversões" dependem do traçado e precisam ser confirmadas diretamente na imagem.
- A rede em `classifier.py` recebe dez contagens de categorias, usa ReLU nas camadas ocultas e softmax na saída. Ainda não foi treinada; precisa de exemplos rotulados representativos e validação antes de gerar previsões.

## Testes

```powershell
python -m unittest
```