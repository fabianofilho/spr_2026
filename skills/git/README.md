# Skill: Git Workflow

## ⚠️ REGRA OBRIGATÓRIA

**SEMPRE commitar após qualquer modificação no repositório!**

---

## 🎯 Quando Ativar

- Após criar/editar qualquer arquivo
- Após criar notebooks
- Após atualizar TODO.md ou insights/
- Após qualquer modificação no repositório

---

## 📋 Checklist

- [ ] Modificações feitas
- [ ] `git add -A`
- [ ] `git commit -m "descrição breve"`
- [ ] Verificar que commit foi criado

---

## 🔧 Comando Padrão

```bash
git add -A && git commit -m "descrição breve"
```

---

## 📝 Padrões de Mensagem

| Tipo | Prefixo | Exemplo |
|------|---------|---------|
| Novo notebook | `feat:` | `feat: add submit_tfidf_xgboost notebook` |
| Correção | `fix:` | `fix: corrigir path do modelo` |
| Documentação | `docs:` | `docs: atualizar leaderboard` |
| Refatoração | `refactor:` | `refactor: reorganizar skills/` |
| Score | `score:` | `score: LinearSVC 0.77885` |

---

## ❌ Por que é Obrigatório?

O Kaggle usa a versão **commitada** do repositório ao fazer upload/commit do notebook.

Se não commitar:
- Kaggle usará versão antiga
- Mudanças não serão refletidas
- Submissão pode falhar ou ter resultados errados

---

## 🔄 Workflow Típico

```bash
# 1. Fazer modificações
# ... editar arquivos ...

# 2. Verificar mudanças
git status

# 3. Commitar
git add -A && git commit -m "feat: descrição"

# 4. Push (se necessário)
git push
```
