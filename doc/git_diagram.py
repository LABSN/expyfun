# -*- coding: utf-8 -*-

import os
from os import path as op

title = 'git flow diagram'

font_face = 'Arial'
node_size = 12
node_small_size = 9
edge_size = 9
local_color = '#7bbeca'
remote_color = '#ff6347'

legend = """
<<FONT POINT-SIZE="%s">
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" CELLPADDING="4">
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">
Local computer</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">
Remote repository</TD></TR>
</TABLE></FONT>>""" % (edge_size, local_color, remote_color)
legend = ''.join(legend.split('\n'))

nodes = dict(
    upstream='LABSN/expyfun.git',
    u1='Eric89GXL/expyfun.git',
    u2='drammock/expyfun.git',
    u1_clone='/home/larsoner/expyun',
    u2_clone='/home/drmccloy/expyun',
    legend=legend,
)

local_space = ('u1_repo', 'u2_repo')
remote_space = ('upstream', 'u1_clone', 'u2_clone')

edges = (
    ('u1_clone', 'u1', 'origin'),
    ('u2_clone', 'u2', 'origin'),
    ('u1_clone', 'upstream', 'upstream'),
    ('u2_clone', 'upstream', 'upstream'),
)

subgraphs = (
    [('upstream', 'u1', 'u2'), ('Internet')],
    [('u1'), ("Eric's computer")],
    [('u2'), ("Dan's computer")],
)


import pygraphviz as pgv
g = pgv.AGraph(name=title, directed=True)

for key, label in nodes.items():
    label = label.split('\n')
    if len(label) > 1:
        label[0] = ('<<FONT POINT-SIZE="%s">' % node_size
                    + label[0] + '</FONT>')
        for li in range(1, len(label)):
            label[li] = ('<FONT POINT-SIZE="%s"><I>' % node_small_size
                         + label[li] + '</I></FONT>')
        label[-1] = label[-1] + '>'
        label = '<BR/>'.join(label)
    else:
        label = label[0]
    g.add_node(key, shape='plaintext', label=label)

# Create and customize nodes and edges
for edge in edges:
    g.add_edge(*edge[:2])
    e = g.get_edge(*edge[:2])
    if len(edge) > 2:
        e.attr['label'] = ('<<I>' +
                           '<BR ALIGN="LEFT"/>'.join(edge[2].split('\n')) +
                           '<BR ALIGN="LEFT"/></I>>')
    e.attr['fontsize'] = edge_size
g.get_node
# Change colors
for these_nodes, color in zip((sensor_space, source_space),
                              (sensor_color, source_color)):
    for node in these_nodes:
        g.get_node(node).attr['fillcolor'] = color
        g.get_node(node).attr['style'] = 'filled'

# Create subgraphs
for si, subgraph in enumerate(subgraphs):
    g.add_subgraph(subgraph[0], 'cluster%s' % si,
                   label=subgraph[1], color='black')

# Format (sub)graphs
for gr in g.subgraphs() + [g]:
    for x in [gr.node_attr, gr.edge_attr]:
        x['fontname'] = font_face
g.node_attr['shape'] = 'box'

# A couple of special ones
for ni, node in enumerate(('fwd', 'inv', 'trans')):
    node = g.get_node(node)
    node.attr['gradientangle'] = 270
    colors = (source_color, sensor_color)
    colors = colors if ni == 0 else colors[::-1]
    node.attr['fillcolor'] = ':'.join(colors)
    node.attr['style'] = 'filled'
del node
g.get_node('legend').attr.update(shape='plaintext', margin=0, rank='sink')
# put legend in same rank/level as inverse
l = g.add_subgraph(['legend', 'inv'], name='legendy')
l.graph_attr['rank'] = 'same'

g.layout('dot')
g.draw('git_flow.svg', format='svg')
