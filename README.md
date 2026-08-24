# Minervini S&P 500 Screener

Ferramenta de screening diário automático que aplica a metodologia SEPA (Specific
Entry Point Analysis) do investidor Mark Minervini a todas as empresas do S&P 500,
e produz sinais de **compra** (novos setups que cumprem o Trend Template + breakout
com volume) e **avisos de venda** (posições que deixaram de cumprir os critérios
técnicos).

⚠️ **Isto não é aconselhamento financeiro.** É uma ferramenta de screening baseada
numa metodologia pública, com heurísticas próprias para o VCP (Volatility
Contraction Pattern) e o RS Rating — não é garantia de resultados. Os dados vêm do
Yahoo Finance (via `yfinance`), uma API não-oficial que ocasionalmente pode falhar.

## O que corre automaticamente

`.github/workflows/daily-screen.yml` corre `minervini_screen.py` todos os dias
úteis às 22:17 UTC (depois do fecho do mercado dos EUA) via GitHub Actions —
gratuito, sem precisares de nenhum servidor. O script:

1. Vai buscar a lista atual de constituintes do S&P 500 (Wikipedia).
2. Descarrega 2 anos de histórico de preços/volume de cada ticker (Yahoo Finance).
3. Calcula os 8 critérios do Trend Template, um RS Rating relativo ao universo,
   e um score heurístico de VCP.
4. Deteta transições nos últimos 10 dias úteis (quem passou a qualificar-se =
   sinal de compra; quem deixou de qualificar-se = aviso de venda).
5. Escreve `signals.json` (para consumo automático) e `report.md` (leitura humana),
   e faz commit dos resultados de volta a este repositório.

## Configuração (uma única vez, ~5 minutos)

1. Cria um repositório novo no GitHub (pode ser **público** — não contém dados
   pessoais nem financeiros teus, só o script e os resultados do screening).
2. Faz upload de todos os ficheiros desta pasta para esse repositório, mantendo a
   estrutura (incluindo a pasta `.github/workflows/`).
3. Vai a **Settings → Actions → General → Workflow permissions** e confirma que
   está marcado **"Read and write permissions"** (necessário para o workflow poder
   fazer commit dos resultados).
4. Vai ao separador **Actions**, escolhe o workflow "Minervini S&P 500 daily
   screen" e corre-o manualmente uma vez (**Run workflow**) para confirmar que
   funciona — demora uns 10-20 minutos a processar as 500 ações.
5. Confirma que `signals.json` e `report.md` foram atualizados no repositório.
6. Envia-me o link do repositório (ex: `https://github.com/o-teu-user/o-teu-repo`)
   — eu configuro o alerta diário automático que lê os resultados e te avisa.

## Ficheiros

- `minervini_screen.py` — o script principal.
- `requirements.txt` — dependências Python.
- `.github/workflows/daily-screen.yml` — agendamento automático (GitHub Actions).
- `signals.json` / `report.md` — resultados do último screening (gerados
  automaticamente; os ficheiros de exemplo aqui incluídos são só placeholders).
- `test_logic.py` — teste offline com dados sintéticos, só para validar a lógica
  (não é necessário para o funcionamento normal).
