# Report del Lab 4 (LaTeX)

Límite de la guía: **3 páginas de texto** (secciones 1–4 + contribuciones) **+ 1 página** de tablas, figuras y referencias.

| Archivo | Contenido |
|---|---|
| `main.tex` | Formato, título y autores; incluye las secciones |
| `sections/1_introduction.tex` | 1. Introducción (contexto, dataset, alcance) |
| `sections/2_methodology.tex` | 2.1 Preprocesado · 2.2 CUSTOM · 2.3 FT · 2.4 Protocolo y selección |
| `sections/3_results.tex` | 3. Resultados y análisis de errores |
| `sections/4_conclusions.tex` | 4. Conclusiones |
| `sections/5_contributions.tex` | Contribuciones individuales (obligatorio) |
| `sections/6_tables_figures.tex` | Página 4: tablas y figuras |
| `references.bib` | Bibliografía (solo salen las citadas) |
| `figures/` | PNG/PDF de las figuras |

- **Compilar:** `latexmk -pdf main.tex` (en Overleaf: subir la carpeta `report/`).
- **Pendiente:** `\todo{...}` marca lo que falta, en rojo; antes de entregar no debe quedar ninguno.
- **Para no pisarse:** cada uno edita su sección y se hace `git pull` antes de empezar.
