# Análise Downstream da Microbiota Intestinal à Mangaba

Código e resultados do trabalho *Downstream Analysis of Gut Microbiome Responses to Mangaba Tea Supplementation*: análise da resposta da microbiota intestinal à polpa e ao extrato de mangaba (*Hancornia speciosa*), a partir de dados de sequenciação do gene 16S rRNA.

A tabela de ASVs anotada foi fornecida pela empresa de sequenciação (Novogene, em QIIME2 com DADA2 e SILVA 138.1). Este repositório contém os passos desenvolvidos a partir daí: correção pelo número de cópias do gene 16S rRNA, normalização e mapeamento funcional.

## Conteúdo

- `scripts/` — os dois scripts em Python:
  - `normalize_copy_number_asv.py`: corrige as ASVs pelo número de cópias do gene 16S rRNA (rrnDB) e normaliza por *Total Sum Scaling*.
  - `fermentation_pathways.py`: mapeia os KO preditos pelo Tax4Fun nas vias de fermentação/SCFA (KEGG).
- `data/` — dados de entrada (tabela de ASVs, números de cópias, metadados, SCFAs/pH, tabelas KEGG).
- `outputs/` — tabelas e figuras geradas pelos scripts.
- `microbiomeanalyst_results/` — composição, diversidade, LEfSe e Tax4Fun, obtidos no [MicrobiomeAnalyst](https://www.microbiomeanalyst.ca/).

## Autoria

Diana Oliveira
