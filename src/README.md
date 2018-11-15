#README

#####Autores
Markus Fischer • Guzmán López • David Pérez • Ander Raso

#####Requisitos
Listados en requirements.txt.

#####Diagrama de clases y módulos
Incluido en class_diagram.png.

####Uso
Para ejecutar cualquier módulo

```python
python3 modulo.py [-h]
```

Con el parámetro -h se mostrará ayuda e información sobre su uso y parámetros.

####Instrucciones
Utilizar el módulo csv_cleaner.py para preprocesar los datos crudos (verbal_autopsies_raw.csv).
```python
python3 csv_cleaner.py verbal_autopsies_raw.csv verbal_autopsies_clean.csv
```
Utilizar el módulo kmeans (o su clase KMeans) para realizar el clustering y guardar los resultados.
```python
python3 kmeans.py -d verbal_autopsies_raw.csv -o output/path/ --text_attribute "open_response" --class_attribute "gs_text34" -k 10 -m 2
```
Utilizar los módulos heatmap.py y plot_sse.py para obtener gráficos representando esos resultados.
```python
python3 heatmap.py results/clusters.csv output/path/heatmap.png
python3 plot_sse.py results/evaluation.csv output/path/
```
