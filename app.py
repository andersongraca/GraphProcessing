import streamlit as st
from graph_utils import Graph, open_graph_txt, MinimumPath, Components


import mpu
import base64
import json
import numpy as np
import pandas as pd
import os

def get_table_download_link_csv(df, filename="file.txt", label="Download file", index=False):
    csv = df.to_csv(index=index).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank">{label}</a>'
    return href

def main():
    st.title("Quarta Prova de Grafos")
    st.header("UFS Universidade Federal de Sergipe")
    menu = ["Carregar grafo", "Representação do grafo", 
   "Caminhos mínimos",]
    choice = st.sidebar.selectbox("Opções", menu)
    
    graph_txt = None
    graph = None

    

    if choice == "Carregar grafo":
        st.subheader("Carregar um grafo através de um arquivo txt")
        st.markdown("""
        <div>
        <p>Faça upload de um arquivo contendo um grafo, onde a primeira linha é o número de vértices, as próximas linhas são as arestas, e a última linha em branco ou com um ponto.</p>
        <p><b>Exemplo:</b></p>
        <p>3<br>1 2<br>2 3<br>.</p>
        </div>
        """, unsafe_allow_html = True)
        graph_txt = st.file_uploader("Upload do Grafo (.txt)", type="txt")
        if graph_txt:
            graph_txt_button = st.button("Ler Grafo")
            if graph_txt_button:
                graph = open_graph_txt(graph_txt)
                graph.sort_neighbors()
                mpu.io.write("graph.pickle", graph)
                st.success(f"Grafo com {graph.n_nodes} nós carregado com sucesso!")
        st.header("Limpar grafo")
        limpar = st.button("Limpar")
        if limpar:
            try:
                os.remove("graph.pickle")
            except:
                pass
            st.success("Grafo excluído com sucesso!")
    
    elif choice == "Representação do grafo":
        st.subheader("Representação do grafo")
        st.markdown("""
        <div>
        <p>Um grafo pode ser tanto representado como uma matriz de adjacência, como listas de adjacência.</p>
        <p>Escolha a opção desejada para baixar o arquivo de representação.</p>
        </div>
        """, unsafe_allow_html = True)
        try:
            graph = mpu.io.read("graph.pickle")
            matriz = st.button("Gerar matriz de adjacência")
            listas = st.button("Gerar listas de adjacência")
            if matriz:
                st.success("Matriz de adjacência gerada com sucesso!")
                st.markdown(get_table_download_link_csv(graph.get_matrix_beautiful(), "matriz.csv", "Download matriz de adjacência (.csv)", index=True), 
                unsafe_allow_html=True)
            if listas:
                lista_json = json.dumps(graph.get_node_edges()).encode()
                b64 = base64.b64encode(lista_json).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="listas.txt" target="_blank">Download listas de adjacência (.txt)</a>'
                
                st.markdown(href, unsafe_allow_html=True)
        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")
    
    elif choice == "Estatísticas":
        st.header("Estatísticas do grafo")
        try:
            graph = mpu.io.read("graph.pickle")
            st.markdown(f"""
            <div>
            <p>Número de pesquisadores: {int(graph.n_nodes)}</p>
            <p>Número de colaborações: {int(graph.get_matrix().sum()/2)}</p>
            <p>Grau mínimo: {int(graph.get_matrix().sum(axis=0).min())}</p>
            <p>Grau máximo: {int(graph.get_matrix().sum(axis=0).max())}</p>
           
            
            </div>
            """, unsafe_allow_html = True)

       