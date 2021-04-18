import json
import os

from graphviz import Digraph
from tensorflow.keras import Model

import re


def node_str(class_name, **kargs):
    attrs_list_str = ["{%s|%s}" % (name, str(value)) for name, value in kargs.items()]
    return "%s|{%s}" % (class_name, "|".join(attrs_list_str))

def add_nodes(g,model,personalized_layers,default_layers,subgraph=False):
    ## Construction du graph graphviz à l'aide des layers keras
    for i,layer in enumerate(model.layers):
        class_name = layer.__class__.__name__
        if class_name == "Functional": # Si on a un sous-modèle
            with g.subgraph(name=f'cluster_{layer.name}' ) as g1:
                add_nodes(g1,layer,personalized_layers,default_layers,True)
            print(model.layers[i-1].name)
            g.edge(model.layers[i-1].name,layer.layers[0].name,label=str(layer.output_shape))
            continue
        attributes_to_show = {}
        ## Choix du fichier json contenant les informations pour afficher ce layer
        ## de préférence celui utilisateur
        if personalized_layers is not None and class_name in personalized_layers.keys():
            dico_access = personalized_layers
        else:
            dico_access = default_layers
        ## Parcourt des attributs et récupération de leurs valeurs
        for attr_name, access in dico_access[class_name]["attributes"].items():
            if access == "same":
                attr_value = eval("layer." + attr_name)
            else:
                attr_value = eval(access)
            if attr_value is not None and attr_value != "NoneType":
                attributes_to_show[attr_name] = attr_value
        ## Création du noeud
        chaine_graphviz = node_str(class_name, **attributes_to_show)
        g.node(layer.name, label=chaine_graphviz, shape="record", **dico_access[class_name]["format"])
        if type(layer.input) is not list and class_name != "InputLayer":
            parent = layer.input.name.split("/")[0]
            parent = parent.split(":")[0]
            parent_layer = model.get_layer(parent)
            if parent_layer.__class__.__name__ == "Functional":
                g.edge(parent_layer.layers[-1].name, layer.name, label=str(layer.input_shape))
            else:
                g.edge(parent, layer.name, label=str(layer.input_shape))

def plot_model(model: Model,output_path,path_personnalizations=None):
    g = Digraph(format="png")
    ## Récupération des fichiers indiquant comment accéder aux attributs et comment formatter le noeud
    ### Fichier par défaut
    regex = re.split(r'\\|/',os.path.realpath(__file__))
    with open("/".join(regex[:-1])+"/layers.json") as f:
        default_layers = json.load(f)
    ### Et si précisé un fichier utilisateur pour éventuellement repréciser certaines couches
    personalized_layers = None
    if path_personnalizations is not None and os.path.exists(path_personnalizations) and path_personnalizations.split(".")[-1] == "json":
        with open("./layers.json") as f:
            personalized_layers = json.load(f)
    add_nodes(g,model,personalized_layers,default_layers)
    try:
        g.save(output_path)
    except:
        pass
    try:
        g.render(output_path)
    except:
        pass