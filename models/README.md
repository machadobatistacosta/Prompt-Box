# 🧠 Models Directory

Este diretório armazena modelos GGUF fine-tunados para uso local.

## Estrutura

```
models/
├── README.md                    # Este arquivo
├── .gitkeep                     # Mantém pasta no Git
├── llama3-juridico-q4.gguf     # Modelo fine-tunado (futuro)
└── custom.gguf                  # Qualquer modelo customizado
```

## Como Adicionar um Modelo

### Opção 1: Via Ollama (Recomendado)

```bash
# Baixar modelo base
ollama pull llama3

# Criar Modelfile para seu modelo fine-tunado
cat > Modelfile << EOF
FROM ./models/llama3-juridico-q4.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
SYSTEM "Você é um assistente jurídico brasileiro especializado."
EOF

# Importar no Ollama
ollama create llama3-juridico -f Modelfile

# Verificar
ollama list
```

### Opção 2: Via llama-cpp-python (Direto)

```bash
# Instalar llama-cpp-python
pip install llama-cpp-python

# Atualizar config.yaml
models:
  local_gguf:
    name: "local"
    provider: "llamacpp"
    gguf_path: "./models/llama3-juridico-q4.gguf"

active_model: "local_gguf"
```

## Modelos Recomendados

### Para Hardware Modesto (8GB RAM)
- `phi3` - Microsoft Phi-3 Mini (3.8B) - via Ollama
- Qualquer GGUF Q4_K_M (4-bit quantized)

### Para Hardware Melhor (16GB+ RAM)
- `llama3` - Meta Llama 3 8B - via Ollama
- GGUF Q8_0 (8-bit quantized)

## Fine-Tuning → GGUF

Após treinar seu modelo no Colab com QLoRA:

```python
# No Colab, após merge do adapter:
model.save_pretrained("merged_model")

# Converter para GGUF
!python llama.cpp/convert-hf-to-gguf.py merged_model --outfile llama3-juridico.gguf

# Quantizar (reduz tamanho)
!./llama.cpp/quantize llama3-juridico.gguf llama3-juridico-q4.gguf Q4_K_M
```

## Arquivos Ignorados

Adicione ao `.gitignore`:
```
models/*.gguf
models/*.bin
```

Modelos GGUF são grandes (2-8GB) e não devem ir pro Git.
