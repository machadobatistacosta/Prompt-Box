"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      PROMPTBOX LOCAL INGEST v2.2                              ║
║             "O Leitor de PDFs/TXTs Sincronizado com o Crawler"                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Features:                                                                    ║
║  - Engine: Mesma lógica Regex do Crawler (compatibilidade total no Banco)     ║
║  - Suporte: PDF (pypdf) e TXT (utf-8)                                         ║
║  - Metadata: Extrai número do artigo para busca exata                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
import os
import glob
import uuid
import re
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

# --- CONFIG ---
COLLECTION_NAME = "promptbox"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PASTA_DOCS = "./docs"     # Coloque seu TCC e PDFs aqui
CAMINHO_BANCO = "./chroma_db"

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ 'pypdf' não instalado. PDFs ignorados. (pip install pypdf)")

def extract_chunks_juridicos(text, source_name):
    """
    Mesma lógica do Crawler: Quebra por 'Art.' usando Regex.
    Garante que arquivos locais e leis da web fiquem iguais no banco.
    """
    if not text: return [], [], []
    
    # Divide preservando o delimitador "Art."
    raw_chunks = re.split(r'(Art\.\s*\d+)', text)
    
    documents = []
    metadatas = []
    ids = []
    
    current_chunk = ""
    
    for segment in raw_chunks:
        # Se for cabeçalho (ex: "Art. 1"), salva o anterior e começa novo
        if re.match(r'Art\.\s*\d+', segment):
            if current_chunk:
                _process_chunk(current_chunk, source_name, documents, metadatas, ids)
            current_chunk = segment
        else:
            current_chunk += segment
            
    if current_chunk:
        _process_chunk(current_chunk, source_name, documents, metadatas, ids)
        
    return documents, metadatas, ids

def _process_chunk(text, source_name, docs, metas, ids):
    if len(text) < 50: return
    
    # Extrai número do artigo
    try:
        match = re.search(r'Art\.\s*([\d\.]+)', text)
        art_num = match.group(1).replace(".", "").strip() if match else "0"
    except:
        art_num = "0"
        
    docs.append(text.strip())
    metas.append({
        "source": source_name,
        "article": art_num,
        "type": "arquivo_local" # Diferencia do que veio do crawler
    })
    ids.append(str(uuid.uuid4()))

def ler_arquivo(filepath):
    """Lê PDF ou TXT."""
    ext = filepath.lower().split('.')[-1]
    
    if ext == 'pdf':
        if not PDF_AVAILABLE: return ""
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"❌ Erro no PDF {filepath}: {e}")
            return ""
            
    elif ext == 'txt':
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Erro no TXT {filepath}: {e}")
            return ""
    
    return ""

def process_jsonl(filepath, embedder, collection):
    """Processa arquivo JSONL (Dataset Jurídico)"""
    print(f"📄 Processando JSONL: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    batch_docs = []
    batch_metas = []
    batch_ids = []
    
    count = 0
    
    for line in tqdm(lines, desc="Chunking JSONL"):
        try:
            data = json.loads(line)
            # Formato esperado: instruction, input, output
            instruction = data.get('instruction', '')
            output = data.get('output', '')
            
            if not instruction or not output:
                continue
                
            # Formata para o banco vetorial
            # Incluir a pergunta e resposta ajuda na busca semântica
            text_content = f"PERGUNTA: {instruction}\nRESPOSTA: {output}"
            
            batch_docs.append(text_content)
            batch_metas.append({
                "source": os.path.basename(filepath),
                "type": "qa_dataset",
                "original_instruction": instruction[:100] # preview
            })
            batch_ids.append(str(uuid.uuid4()))
            
            # Batch process
            if len(batch_docs) >= 100:
                embeddings = embedder.encode(batch_docs)
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings.tolist()
                )
                count += len(batch_docs)
                batch_docs = []
                batch_metas = []
                batch_ids = []
                
        except json.JSONDecodeError:
            continue
            
    # Processa restantes
    if batch_docs:
        embeddings = embedder.encode(batch_docs)
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings.tolist()
        )
        count += len(batch_docs)
        
    return count

def main():
    print("💉 PromptBox Local Ingest v2.2")
    
    # 1. Banco (Não deleta, apenas conecta para ADICIONAR aos dados do crawler)
    client = chromadb.PersistentClient(path=CAMINHO_BANCO)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    
    # 2. Arquivos
    if not os.path.exists(PASTA_DOCS):
        os.makedirs(PASTA_DOCS)
        print(f"📁 Pasta '{PASTA_DOCS}' criada. Coloque arquivos lá.")
        return

    arquivos = glob.glob(os.path.join(PASTA_DOCS, "*.*"))
    arquivos = [f for f in arquivos if f.lower().endswith(('.pdf', '.txt'))]
    
    # Verifica JSONL na raiz (específico do caso do usuário)
    jsonl_root = glob.glob("*.jsonl")
    arquivos.extend(jsonl_root)
    
    # Verifica JSONL na pasta docs
    jsonl_docs = glob.glob(os.path.join(PASTA_DOCS, "*.jsonl"))
    arquivos.extend(jsonl_docs)
    
    # Remove duplicatas
    arquivos = list(set(arquivos))
    
    if not arquivos:
        print("⚠️  Nenhum arquivo PDF, TXT ou JSONL encontrado.")
        return

    print(f"📂 Processando {len(arquivos)} arquivos locais...")
    
    count = 0
    for arquivo in tqdm(arquivos):
        nome = os.path.basename(arquivo)
        
        # Tratamento especial para JSONL
        if nome.lower().endswith('.jsonl'):
            added = process_jsonl(arquivo, embedder, collection)
            count += added
            continue
            
        texto = ler_arquivo(arquivo)
        
        docs, metas, ids = extract_chunks_juridicos(texto, nome)
        
        if docs:
            # Batch
            batch_size = 100
            for i in range(0, len(docs), batch_size):
                end = i + batch_size
                batch_emb = embedder.encode(docs[i:end])
                
                collection.add(
                    ids=ids[i:end],
                    documents=docs[i:end],
                    metadatas=metas[i:end],
                    embeddings=batch_emb.tolist()
                )
            count += len(docs)
            
    print(f"✅ Sucesso! {count} novos fragmentos adicionados ao Banco.")

if __name__ == "__main__":
    main()