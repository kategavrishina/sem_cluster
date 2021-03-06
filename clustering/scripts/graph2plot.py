import sys
import networkx as nx
from modules import get_clusters, get_data_maps_nodes
from plotting_ import plot_graph_interactive, plot_graph_static
import csv
from itertools import combinations
import os


[_, input_file, output_folder, color, mode, style, edge_label_style, annotators, threshold, is_spring, modus] = sys.argv


threshold=float(threshold)
graph = nx.read_gpickle(input_file)
name = graph.graph['lemma']
try:
    clusters = get_clusters(graph)
except KeyError:
    print('no clusters found')
    clusters = [{n for n in graph.nodes()}]
#normalization = lambda x: (x*3)+1.0
#normalization = lambda x: x+threshold
normalization = lambda x: x
with open(annotators, encoding='utf-8') as csvfile: 
    reader = csv.DictReader(csvfile, delimiter='\t',quoting=csv.QUOTE_NONE,strict=True)
    annotators = [row['annotator'] for row in reader]

if modus=='test':
    dpi = 5
if modus=='system':
    dpi = 5
if modus=='full':
    dpi = 300

if is_spring=='True':
    is_spring = True
if is_spring=='False':
    is_spring = False

output_folder_full = output_folder + '/full/'    
if not os.path.exists(output_folder_full):
    os.makedirs(output_folder_full)    

if style=='interactive':
    plot_graph_interactive(graph, output_folder_full + name, clusters, threshold=threshold, normalization = normalization, period='full', color=color, mode=mode, edge_label_style = edge_label_style, annotators = annotators, is_spring = is_spring)
if style=='static':
    plot_graph_static(graph, output_folder_full + name, clusters, threshold=threshold, normalization = normalization, period='full', color=color, mode=mode, edge_label_style = edge_label_style,annotators = annotators, dpi=dpi, is_spring = is_spring, node_size=300)
    
mappings_nodes = get_data_maps_nodes(graph)
node2period = mappings_nodes['node2period']
periods = sorted(set(node2period.values()))
if len(periods) > 1:
    for period in periods:

        output_folder_period = output_folder + '/' + period + '/'
        if not os.path.exists(output_folder_period):
            os.makedirs(output_folder_period)    

        if style=='interactive':
            plot_graph_interactive(graph, output_folder_period + name, clusters, threshold = threshold, normalization = normalization, period=period, color=color, mode=mode, edge_label_style = edge_label_style,annotators = annotators, is_spring = is_spring)
        if style=='static':
            plot_graph_static(graph, output_folder_period + name, clusters, threshold = threshold, normalization = normalization, period=period, color=color, mode=mode, edge_label_style = edge_label_style, annotators = annotators, dpi=dpi, is_spring = is_spring, node_size=300) 

    if style=='interactive':
        output_folder_aligned = output_folder+'/aligned/'    
        if not os.path.exists(output_folder_aligned):
            os.makedirs(output_folder_aligned)    
        combos = combinations(periods, 2)
        for (old, new) in combos:
            with open(output_folder_aligned + name + '_{0}_{1}'.format(old, new) + '.html', 'w', encoding='utf-8') as f_out:
                f_out.write("<html>\n<head>\n</head>\n<frameset cols=\"50\%,\*\">\n<frame src=\"../{0}/{1}\">\n<frame src=\"../{2}/{1}\">\n</frameset>\n</html>\n".format(old, name+'.html', new))

