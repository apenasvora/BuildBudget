# == OBJETIVO GERAL ==
Criar um MVP que gera, atualiza e monitora orçamentos de obras de forma automática e em tempo real, combinando:
1. Extração de quantidades a partir de modelos BIM/CAD.
2. Algoritmos de IA (ML + visão computacional) para prever consumo de materiais e ajustar custos.
3. Dashboard web com feedback instantâneo e inputs manuais.
4. Metodologias ágeis (Lean Startup + Design Thinking) para iterar rapidamente.

# == ENTREGÁVEIS DO MVP ==
• API REST/Python (FastAPI) → endpoints para upload do modelo BIM/CAD, upload de imagens do canteiro e leitura da base de custos.  
• Worker de IA → scripts de ML que:
   – analisam histórico (CSV) + progresso da obra,  
   – aplicam visão computacional (OpenCV + YOLO, opcional) para contar peças/pallets,  
   – recalculam o orçamento e salvam em banco (PostgreSQL/SQLite).  
• Front-end (React + Vite ou simples HTML/Tailwind) → dashboard com:
   – cards de KPI (custo previsto × real, % atraso, top desvios),  
   – gráfico de linha “custo acumulado” live,  
   – formulário para ajustes manuais.  
• Script de seed → popula base com normas-chave (p. ex. NBR 12721) e tabelas de composição de custos.  
• README → passo a passo para rodar/duplicar o Replit, estruturar variáveis de ambiente e executar testes.

# == BACKLOG DE FUNCIONALIDADES ==
1. **Importador BIM/CAD**  
   - Aceitar .IFC (preferencial), .DWG (beta).  
   - Mapear elementos → classes de materiais e quantidades.

2. **Motor de Previsão & Ajuste**  
   - `train.py` → regressão/XGBoost usando histórico de consumo.  
   - `predict.py` → gera forecast diário; aplica fator de correção se cronograma mudou.  
   - `vision.py` (opcional) → detecta materiais nas fotos; compara com previsto.

3. **Orquestração de Eventos**  
   - Webhook “modelo novo” → reprocessa orçamento.  
   - Cron job (cada 6 h) → recalcula tendências e envia alertas.

4. **Interface Web**  
   - Modo “Timeline” (custo ao longo do tempo).  
   - Modo “Heat-map” de desvios por categoria de material.  
   - Botão “Aceitar ajuste” para travar orçamento revisto.

5. **Conformidade & Normas**  
   - Valida coeficientes contra NBR 12721; loga divergências.  
   - Exporta relatório PDF “Memorial de Cálculo”.

# == METODOLOGIA DE DESENVOLVIMENTO ==
• **Sprint 0** (1 dia): setup Replit, definições de stack, importador .IFC mínimo.  
• **Sprint 1** (3 dias): parsing BIM → quantidades, API básica, dashboard placeholder.  
• **Sprint 2** (3 dias): modelo de previsão + atualização dinâmica + gráfico vivo.  
• **Sprint 3** (2 dias): visão computacional opcional, validação NBR, polish UI.  
• **Feedback loop** ao fim de cada sprint com 2 usuários (arquiteto + engenheiro).  

# == CRITÉRIOS DE SUCESSO ==
✓ Custo previsto recalcular em ≤ 5 s após upload de novo modelo.  
✓ Dashboard mostra desvios em tempo real (latência máx. 2 s via WebSocket).  
✓ Precisão da previsão ≥ 90 % para top-5 materiais (MAPE).  
✓ Código 100 % replicável em Replit sem ajustes locais.  
✓ Documentação enxuta: README + passo a passo de deploy.

# == TECNOLOGIAS SUGERIDAS ==
| Camada          | Stack no Replit                      |
|-----------------|--------------------------------------|
| Backend API     | Python 3.12 + FastAPI                |
| ML / Visão      | scikit-learn, XGBoost, OpenCV, YOLOv8|
| Parser BIM      | `ifcopenshell`                       |
| Banco de dados  | SQLite (dev) → PostgreSQL (prod)     |
| Front-end       | React + Vite + Tailwind (ou HTMX)    |
| Realtime layer  | FastAPI WebSockets ou Socket.io      |
| Auth (futuro)   | Clerk.dev ou Firebase Auth           |

# == INSTRUÇÕES RÁPIDAS PARA RODAR NO REPLIT ==
