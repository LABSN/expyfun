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
Local computers</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">
Remote repositories</TD></TR>
</TABLE></FONT>>""" % (edge_size, local_color, remote_color)
legend = ''.join(legend.split('\n'))

nodes = dict(
    upstream='LABSN/expyfun\n'
             'master\n'
             ' ',
    maint='Eric89GXL/expyfun\n'
          'master\n'
          'other_branch',
    dev='rkmaddox/expyfun\n'
        'master\n'
        'fix_branch',
    maint_clone='/home/larsoner/expyfun\n'
                'master (origin/master)\n'
                'other_branch (origin/other_branch)\n'
                'ross_branch (rkmaddox/fix_branch)',
    dev_clone='/home/rkmaddox/expyfun\n'
              'master (origin/master)\n'
              'fix_branch (origin/fix_branch)\n'
              ' ',
    user_clone='/home/akclee/expyfun\n'
               'master (origin/master)\n'
               ' \n'
               ' ',
    legend=legend,
)

remote_space = ('maint', 'dev', 'upstream')
local_space = ('maint_clone', 'dev_clone', 'user_clone')

edges = (
    ('maint_clone', 'maint', 'origin'),
    ('dev_clone', 'dev', 'origin'),
    ('user_clone', 'upstream', 'origin'),
    ('maint_clone', 'upstream', 'upstream'),
    ('maint_clone', 'dev', 'rkmaddox'),
    ('dev_clone', 'upstream', 'upstream'),
)

subgraphs = (
    [('upstream', 'maint', 'dev'), ('GitHub')],
    [('maint_clone'), ('Maintainer')],
    [('dev_clone'), ("Developer")],
    [('user_clone'), ("User")],
)


import pygraphviz as pgv
g = pgv.AGraph(name=title, directed=True)

for key, label in nodes.items():
    label = label.split('\n')
    if len(label) > 1:
        label[0] = ('<<FONT POINT-SIZE="%s"><B>' % node_size
                    + label[0] + '</B></FONT>')
        for li in range(1, len(label)):
            label[li] = ('<FONT POINT-SIZE="%s"><I>' % node_small_size
                         + label[li] + '</I></FONT>')
        label[-1] = label[-1] + '<BR ALIGN="LEFT"/>>'
        label = '<BR ALIGN="LEFT"/>'.join(label)
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
for these_nodes, color in zip((local_space, remote_space),
                              (local_color, remote_color)):
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

g.get_node('legend').attr.update(shape='plaintext', margin=0, rank='sink')
# put legend in same rank/level as inverse
l = g.add_subgraph(['legend', 'inv'], name='legendy')
l.graph_attr['rank'] = 'same'

g.layout('dot')
g.draw('git_flow.svg', format='svg')
