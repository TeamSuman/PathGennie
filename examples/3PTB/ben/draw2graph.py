#!/usr/bin/python

import yaml

import sys
import os
import math

from networkx.utils import open_file
try:
    import cPickle as pickle
except ImportError:
    import pickle
from networkx.algorithms import isomorphism
import networkx as nx

@open_file(0, mode='rb')
def read_gpickle(path):
    return pickle.load(path)

def CalcDihe(x,y,z):
    x1 = x[0] - x[1]
    y1 = y[0] - y[1]
    z1 = z[0] - z[1]
    u1 = x[2] - x[1]
    v1 = y[2] - y[1]
    w1 = z[2] - z[1]
    x2 = x[1] - x[2]
    y2 = y[1] - y[2]
    z2 = z[1] - z[2]
    u2 = x[3] - x[2]
    v2 = y[3] - y[2]
    w2 = z[3] - z[2]

    a1 = y1 * w1 - z1 * v1
    b1 = z1 * u1 - x1 * w1
    c1 = x1 * v1 - y1 * u1

    a2 = y2 * w2 - z2 * v2
    b2 = z2 * u2 - x2 * w2
    c2 = x2 * v2 - y2 * u2

    dot = a1 * a2 + b1 * b2 + c1 * c2
    sa1 = a1 * a1
    sb1 = b1 * b1
    sc1 = c1 * c1
    sa2 = a2 * a2
    sb2 = b2 * b2
    sc2 = c2 * c2

    sq1 = sa1 + sb1 + sc1
    sq2 = sa2 + sb2 + sc2

    rt1 = math.sqrt(sq1)
    rt2 = math.sqrt(sq2)
    den = rt1 * rt2
    cos = dot / den

    rad = math.acos(cos)

    aphi = rad * 180 / math.pi

    i = y1 * w2 - z1 * v2
    j = z1 * u2 - x1 * w2
    k = x1 * v2 - y1 * u2

    f = i * u1 + j * v1 + k * w1

    if f > 0:
        phi =  aphi
    if f < 0:
        phi = -aphi
    return phi

if __name__ == '__main__':

    archive = sys.argv[1]
    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    h = nx.Graph()
    serial = 1
    for line in open('drawing_3D.mol'):
        cnt = len(line.split())
        if cnt == 16:
            x, y, z, element, massdiff, charge = line.strip().split()[:6]
            h.add_node(int(serial), **{'x' : float(x), 'y': float(y), 'z': float(z),
            'element': element, 'massdiff': massdiff, 'charge': charge})
            serial += 1
            continue
        if (cnt==6 or cnt == 7) and not line.startswith('M') and not 'V' in line:
            atomi = int(line[0:3])
            atomj = int(line[3:6])
            h.add_edge(atomi, atomj)

    nx.write_gpickle(h,"drawing.gpickle")

    rmH = []

    for node in h.nodes():
        if h.node[node]['element'] == 'H':
            rmH.append(node)

    for node in rmH:
        h.remove_node(node)

    nx.write_gpickle(h,"drawing_no_h.gpickle")

    exact_resid = []
    iso_resid = []
    proto_resid = []
    score_lib = {}
    for filename in os.listdir("%s/ligandrm/no_h/total" % archive):
        resname = filename.split('.')[0]
        h = nx.read_gpickle("drawing_no_h.gpickle")
        g = nx.read_gpickle("%s/ligandrm/no_h/total/%s" % (archive,filename))

        GM = isomorphism.GraphMatcher(h,g)
        for mapping1 in GM.match():
            matched1 = True
            for k in mapping1:
                hatom = k
                gatom = mapping1[k]
                if h.node[hatom]['element'] != g.node[gatom]['element']:
                    matched1 = False
                    break
            if matched1:
                break
        if GM.is_isomorphic():
            h = nx.read_gpickle("drawing.gpickle")
            g = nx.read_gpickle("%s/ligandrm/with_h/total/%s" % (archive,filename))
            pdb_path = "%s/csml/%s.pdb" % (archive,filename.split('.')[0].lower())
            if not os.path.exists(pdb_path):
                if verbose:
                    print("Skipping missing PDB file:", filename)
                continue
            pdbopen = open(pdb_path)
            for line in pdbopen:
                if line.startswith('ATOM'):
                    serial     = int(line[6:11])
                    atomn      = line[12:16].strip()
                    resn       = line[17:20].strip()
                    chainid    = line[21]
                    x          = float(line[30:38])
                    y          = float(line[38:46])
                    z          = float(line[46:54])
                    for n,data in g.nodes(True):
                        if data['atomname'] == atomn:
                            g.add_node(int(n), **{'x': float(x), 'y': float(y), 'z': float(z)})
            GM = isomorphism.GraphMatcher(h,g)
            if GM.is_isomorphic():
                for mapping in GM.match():
                    matched = True
                    for k in mapping:
                        hatom = k
                        gatom = mapping[k]
                        if h.node[hatom]['element'] != g.node[gatom]['element']:
                            matched = False
                            break
                    if matched:
                        break
                if matched:
                    for node, data in h.nodes(True):
                        if matched and h.degree(node) == 4 and h.node[node]['element'] == 'C':
                            subgraphs1 = [n for n in h[node]]
                            subgraphs2 = [mapping[n] for n in h[node]]
                            atomtypes1 = [h.node[n]['element'] for n in subgraphs1]
                            atomtypes2 = [g.node[n]['element'] for n in subgraphs2]
                            if atomtypes1.count('H') == 1:
                                x = [h.node[n]['x'] for n in subgraphs1]
                                y = [h.node[n]['y'] for n in subgraphs1]
                                z = [h.node[n]['z'] for n in subgraphs1]
                                phi = CalcDihe(x,y,z)
                                x1 = [g.node[n]['x'] for n in subgraphs2]
                                y1 = [g.node[n]['y'] for n in subgraphs2]
                                z1 = [g.node[n]['z'] for n in subgraphs2]
                                phi1 = CalcDihe(x1,y1,z1)
                                if phi * phi1 > 0:
                                    matched = True
                                else:
                                    matched = False
                    if matched:
                        exact_resid.append(resname)
                    else:
                        iso_resid.append(resname)
            elif matched1:
                proto_resid.append(resname)

    stream = open('csml_results.yml','w')

    csml = {}
    csml["exact"] = exact_resid
    csml["iso"]   = iso_resid
    csml["proto"] = proto_resid

    yaml.dump(csml, stream, default_flow_style=None)
