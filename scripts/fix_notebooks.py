#!/usr/bin/env python3
import json
import os

# Template de verificação de dataset
DATASET_CHECK = [
    '\n',
    '# ========== VERIFICAR DATASET PRIMEIRO ==========\n',
    'if not os.path.exists(DATA_DIR):\n',
    '    print("\\n" + "="*60)\n',
    '    print("ERRO: Dataset nao encontrado!")\n',
    '    print("="*60)\n',
    '    print("\\nAdicione o dataset:")\n',
    '    print("Add Input -> Competition -> spr-2026-mammography-report-classification")\n',
    '    raise FileNotFoundError(f"Dataset nao encontrado: {DATA_DIR}")\n',
    'print(f"Dataset: {DATA_DIR}")\n',
]

# Template de instalação bitsandbytes (para notebooks com USE_4BIT)
BITSANDBYTES_CHECK = [
    '\n',
    '# Instalar bitsandbytes para 4-bit\n',
    'if USE_4BIT:\n',
    '    import subprocess\n',
    '    subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1"], check=True)\n',
    '    print("bitsandbytes instalado!")\n',
]

def fix_notebook(filepath, add_bitsandbytes=False):
    """Adiciona verificação de dataset após DATA_DIR"""
    try:
        with open(filepath, 'r') as f:
            nb = json.load(f)
        
        modified = False
        
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            
            source = cell['source']
            src_text = ''.join(source)
            
            # Verificar se já tem a verificação
            if 'VERIFICAR DATASET PRIMEIRO' in src_text:
                print(f'  ✓ Já tem verificação: {os.path.basename(filepath)}')
                return False
            
            # Encontrar linha com DATA_DIR
            new_source = []
            found = False
            
            for i, line in enumerate(source):
                new_source.append(line)
                
                # Após linha com DATA_DIR = '...', adicionar verificação
                if "DATA_DIR = '/kaggle/input/" in line and not found:
                    found = True
                    # Adicionar verificação
                    new_source.extend(DATASET_CHECK)
                    
                    # Se tem USE_4BIT, adicionar bitsandbytes
                    if add_bitsandbytes and 'USE_4BIT' in src_text:
                        new_source.extend(BITSANDBYTES_CHECK)
            
            if found:
                cell['source'] = new_source
                modified = True
                break  # Só primeira célula com DATA_DIR
        
        if modified:
            with open(filepath, 'w') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f'  ✓ Corrigido: {os.path.basename(filepath)}')
            return True
        else:
            print(f'  - DATA_DIR não encontrado: {os.path.basename(filepath)}')
            return False
    except Exception as e:
        print(f'  ✗ Erro: {os.path.basename(filepath)} - {e}')
        return False

# Notebooks para corrigir
base = '/Users/f.filho/Projetos/kaggle/spr_2026/resubmit'
notebooks = [
    # 2026-02-27
    (f'{base}/2026-02-27/resubmit_lgbm_v3.ipynb', False),
    (f'{base}/2026-02-27/resubmit_mistral_oneshot.ipynb', True),
    (f'{base}/2026-02-27/resubmit_phi35_oneshot.ipynb', False),
    (f'{base}/2026-02-27/resubmit_qwen25_oneshot.ipynb', True),
    (f'{base}/2026-02-27/resubmit_super_ensemble_v1.ipynb', False),
    # 2026-02-28
    (f'{base}/2026-02-28/resubmit_bertimbau_v4.ipynb', False),
    (f'{base}/2026-02-28/resubmit_ensemble_v3.ipynb', False),
    (f'{base}/2026-02-28/resubmit_gemma3_oneshot.ipynb', False),
    (f'{base}/2026-02-28/resubmit_linearsvc_v4.ipynb', False),
    (f'{base}/2026-02-28/resubmit_sgd_v4.ipynb', False),
]

print('Corrigindo notebooks...\n')
count = 0
for path, add_bnb in notebooks:
    if os.path.exists(path):
        if fix_notebook(path, add_bnb):
            count += 1
    else:
        print(f'  - Arquivo não existe: {os.path.basename(path)}')

print(f'\n{count} notebooks corrigidos!')
