# Let's generate four reference files for the user:
# 1) A Microsoft Word Bibliography XML (sources.xml) that can be imported via References > Manage Sources.
# 2) A BibTeX file (references.bib)
# 3) A RIS file (references.ris)
# 4) A CSL-JSON file (references.csl.json) for Zotero/Mendeley/etc.

word_xml = """<?xml version="1.0" encoding="UTF-8"?>
<b:Sources xmlns:b="http://schemas.openxmlformats.org/officeDocument/2006/bibliography">
  <b:Source>
    <b:Tag>RobbinsMonro1951</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Robbins</b:Last><b:First>Herbert</b:First></b:Person><b:Person><b:Last>Monro</b:Last><b:First>Sutton</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>A Stochastic Approximation Method</b:Title>
    <b:JournalName>Annals of Mathematical Statistics</b:JournalName>
    <b:Year>1951</b:Year>
    <b:Volume>22</b:Volume>
    <b:Pages>400-407</b:Pages>
  </b:Source>
  <b:Source>
    <b:Tag>Polyak1964</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Polyak</b:Last><b:First>Boris T.</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Some methods of speeding up the convergence of iteration methods</b:Title>
    <b:JournalName>USSR Computational Mathematics and Mathematical Physics</b:JournalName>
    <b:Year>1964</b:Year>
    <b:Volume>4</b:Volume>
    <b:Pages>1-17</b:Pages>
  </b:Source>
  <b:Source>
    <b:Tag>Nesterov1983</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Nesterov</b:Last><b:First>Yurii</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)</b:Title>
    <b:JournalName>Soviet Mathematics Doklady</b:JournalName>
    <b:Year>1983</b:Year>
    <b:Volume>27</b:Volume>
    <b:Pages>372-376</b:Pages>
  </b:Source>
  <b:Source>
    <b:Tag>KingmaBa2015</b:Tag>
    <b:SourceType>ConferenceProceedings</b:SourceType>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Kingma</b:Last><b:First>Diederik P.</b:First></b:Person><b:Person><b:Last>Ba</b:Last><b:First>Jimmy</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Adam: A Method for Stochastic Optimization</b:Title>
    <b:ConferenceName>International Conference on Learning Representations (ICLR)</b:ConferenceName>
    <b:Year>2015</b:Year>
    <b:City>San Diego</b:City>
    <b:URL>https://arxiv.org/abs/1412.6980</b:URL>
  </b:Source>
  <b:Source>
    <b:Tag>Andrychowicz2016</b:Tag>
    <b:SourceType>ConferenceProceedings</b:SourceType>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Andrychowicz</b:Last><b:First>Marcin</b:First></b:Person>
      <b:Person><b:Last>Denil</b:Last><b:First>Misha</b:First></b:Person>
      <b:Person><b:Last>Gomez</b:Last><b:First>Sergio</b:First></b:Person>
      <b:Person><b:Last>Hoffman</b:Last><b:First>Matthew</b:First></b:Person>
      <b:Person><b:Last>Pfau</b:Last><b:First>David</b:First></b:Person>
      <b:Person><b:Last>Schaul</b:Last><b:First>Tom</b:First></b:Person>
      <b:Person><b:Last>de Freitas</b:Last><b:First>Nando</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Learning to learn by gradient descent by gradient descent</b:Title>
    <b:ConferenceName>Advances in Neural Information Processing Systems (NeurIPS)</b:ConferenceName>
    <b:Year>2016</b:Year>
  </b:Source>
  <b:Source>
    <b:Tag>Wichrowska2017</b:Tag>
    <b:SourceType>ConferenceProceedings</b:SourceType>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Wichrowska</b:Last><b:First>Olga</b:First></b:Person>
      <b:Person><b:Last>Maheswaranathan</b:Last><b:First>Niru</b:First></b:Person>
      <b:Person><b:Last>Hoffman</b:Last><b:First>Matthew W.</b:First></b:Person>
      <b:Person><b:Last>Grezes</b:Last><b:First>Sergey</b:First></b:Person>
      <b:Person><b:Last>Denil</b:Last><b:First>Misha</b:First></b:Person>
      <b:Person><b:Last>de Freitas</b:Last><b:First>Nando</b:First></b:Person>
      <b:Person><b:Last>Sohl-Dickstein</b:Last><b:First>Jascha</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Learned Optimizers that Scale and Generalize</b:Title>
    <b:ConferenceName>International Conference on Machine Learning (ICML)</b:ConferenceName>
    <b:Year>2017</b:Year>
  </b:Source>
  <b:Source>
    <b:Tag>StanleyMiikkulainen2002</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Stanley</b:Last><b:First>Kenneth O.</b:First></b:Person>
      <b:Person><b:Last>Miikkulainen</b:Last><b:First>Risto</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Evolving neural networks through augmenting topologies</b:Title>
    <b:JournalName>Evolutionary Computation</b:JournalName>
    <b:Year>2002</b:Year>
    <b:Volume>10</b:Volume>
    <b:Issue>2</b:Issue>
    <b:Pages>99-127</b:Pages>
  </b:Source>
  <b:Source>
    <b:Tag>Holland1975</b:Tag>
    <b:SourceType>Book</b:SourceType>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Holland</b:Last><b:First>John H.</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Adaptation in Natural and Artificial Systems</b:Title>
    <b:Year>1975</b:Year>
    <b:City>Ann Arbor</b:City>
    <b:Publisher>University of Michigan Press</b:Publisher>
  </b:Source>
  <b:Source>
    <b:Tag>Salimans2017</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Salimans</b:Last><b:First>Tim</b:First></b:Person>
      <b:Person><b:Last>Ho</b:Last><b:First>Jonathan</b:First></b:Person>
      <b:Person><b:Last>Chen</b:Last><b:First>Xiang</b:First></b:Person>
      <b:Person><b:Last>Sutskever</b:Last><b:First>Ilya</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Evolution Strategies as a Scalable Alternative to Reinforcement Learning</b:Title>
    <b:JournalName>arXiv preprint arXiv:1703.03864</b:JournalName>
    <b:Year>2017</b:Year>
    <b:URL>https://arxiv.org/abs/1703.03864</b:URL>
  </b:Source>
  <b:Source>
    <b:Tag>KirschSchmidhuber2021</b:Tag>
    <b:SourceType>ConferenceProceedings</b:SourceType>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Kirsch</b:Last><b:First>Louis</b:First></b:Person>
      <b:Person><b:Last>Schmidhuber</b:Last><b:First>Jürgen</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Meta Learning Backpropagation and Improving It</b:Title>
    <b:ConferenceName>NeurIPS</b:ConferenceName>
    <b:Year>2021</b:Year>
    <b:URL>https://arxiv.org/abs/2012.14905</b:URL>
  </b:Source>
</b:Sources>
"""

bibtex = r"""
@article{robbins1951stochastic,
  title={A Stochastic Approximation Method},
  author={Robbins, Herbert and Monro, Sutton},
  journal={Annals of Mathematical Statistics},
  volume={22},
  pages={400--407},
  year={1951}
}

@article{polyak1964some,
  title={Some methods of speeding up the convergence of iteration methods},
  author={Polyak, Boris T.},
  journal={USSR Computational Mathematics and Mathematical Physics},
  volume={4},
  pages={1--17},
  year={1964}
}

@article{nesterov1983method,
  title={A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)},
  author={Nesterov, Yurii},
  journal={Soviet Mathematics Doklady},
  volume={27},
  pages={372--376},
  year={1983}
}

@inproceedings{kingma2015adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  booktitle={International Conference on Learning Representations},
  year={2015},
  url={https://arxiv.org/abs/1412.6980}
}

@inproceedings{andrychowicz2016learning,
  title={Learning to learn by gradient descent by gradient descent},
  author={Andrychowicz, Marcin and Denil, Misha and Gomez, Sergio and Hoffman, Matthew and Pfau, David and Schaul, Tom and de Freitas, Nando},
  booktitle={Advances in Neural Information Processing Systems},
  year={2016}
}

@inproceedings{wichrowska2017learned,
  title={Learned Optimizers that Scale and Generalize},
  author={Wichrowska, Olga and Maheswaranathan, Niru and Hoffman, Matthew W and Colmenarejo, Sergio G{\'o}mez and Denil, Misha and de Freitas, Nando and Sohl-Dickstein, Jascha},
  booktitle={International Conference on Machine Learning},
  year={2017}
}

@article{stanley2002evolving,
  title={Evolving neural networks through augmenting topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary Computation},
  volume={10},
  number={2},
  pages={99--127},
  year={2002}
}

@book{holland1975adaptation,
  title={Adaptation in Natural and Artificial Systems},
  author={Holland, John H},
  year={1975},
  publisher={University of Michigan Press}
}

@article{salimans2017evolution,
  title={Evolution Strategies as a Scalable Alternative to Reinforcement Learning},
  author={Salimans, Tim and Ho, Jonathan and Chen, Xiang and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1703.03864},
  year={2017},
  url={https://arxiv.org/abs/1703.03864}
}

@inproceedings{kirsch2021meta,
  title={Meta Learning Backpropagation and Improving It},
  author={Kirsch, Louis and Schmidhuber, J{\"u}rgen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021},
  url={https://arxiv.org/abs/2012.14905}
}
"""

ris = r"""
TY  - JOUR
AU  - Robbins, Herbert
AU  - Monro, Sutton
TI  - A Stochastic Approximation Method
JO  - Annals of Mathematical Statistics
VL  - 22
SP  - 400
EP  - 407
PY  - 1951
ER  - 

TY  - JOUR
AU  - Polyak, Boris T.
TI  - Some methods of speeding up the convergence of iteration methods
JO  - USSR Computational Mathematics and Mathematical Physics
VL  - 4
SP  - 1
EP  - 17
PY  - 1964
ER  - 

TY  - JOUR
AU  - Nesterov, Yurii
TI  - A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)
JO  - Soviet Mathematics Doklady
VL  - 27
SP  - 372
EP  - 376
PY  - 1983
ER  - 

TY  - CPAPER
AU  - Kingma, Diederik P.
AU  - Ba, Jimmy
TI  - Adam: A Method for Stochastic Optimization
T2  - International Conference on Learning Representations (ICLR)
PY  - 2015
UR  - https://arxiv.org/abs/1412.6980
ER  - 

TY  - CPAPER
AU  - Andrychowicz, Marcin
AU  - Denil, Misha
AU  - Gomez, Sergio
AU  - Hoffman, Matthew
AU  - Pfau, David
AU  - Schaul, Tom
AU  - de Freitas, Nando
TI  - Learning to learn by gradient descent by gradient descent
T2  - Advances in Neural Information Processing Systems (NeurIPS)
PY  - 2016
ER  - 

TY  - CPAPER
AU  - Wichrowska, Olga
AU  - Maheswaranathan, Niru
AU  - Hoffman, Matthew W.
AU  - Gomez Colmenarejo, Sergio
AU  - Denil, Misha
AU  - de Freitas, Nando
AU  - Sohl-Dickstein, Jascha
TI  - Learned Optimizers that Scale and Generalize
T2  - International Conference on Machine Learning (ICML)
PY  - 2017
ER  - 

TY  - JOUR
AU  - Stanley, Kenneth O.
AU  - Miikkulainen, Risto
TI  - Evolving neural networks through augmenting topologies
JO  - Evolutionary Computation
VL  - 10
IS  - 2
SP  - 99
EP  - 127
PY  - 2002
ER  - 

TY  - BOOK
AU  - Holland, John H.
TI  - Adaptation in Natural and Artificial Systems
PB  - University of Michigan Press
CY  - Ann Arbor
PY  - 1975
ER  - 

TY  - JOUR
AU  - Salimans, Tim
AU  - Ho, Jonathan
AU  - Chen, Xiang
AU  - Sutskever, Ilya
TI  - Evolution Strategies as a Scalable Alternative to Reinforcement Learning
JO  - arXiv preprint arXiv:1703.03864
PY  - 2017
UR  - https://arxiv.org/abs/1703.03864
ER  - 

TY  - CPAPER
AU  - Kirsch, Louis
AU  - Schmidhuber, Jürgen
TI  - Meta Learning Backpropagation and Improving It
T2  - Advances in Neural Information Processing Systems (NeurIPS)
PY  - 2021
UR  - https://arxiv.org/abs/2012.14905
ER  - 
"""

csl_json = [
  {
    "id": "RobbinsMonro1951",
    "type": "article-journal",
    "title": "A Stochastic Approximation Method",
    "author": [{"family": "Robbins", "given": "Herbert"}, {"family": "Monro", "given": "Sutton"}],
    "container-title": "Annals of Mathematical Statistics",
    "volume": "22",
    "page": "400-407",
    "issued": {"date-parts": [[1951]]}
  },
  {
    "id": "Polyak1964",
    "type": "article-journal",
    "title": "Some methods of speeding up the convergence of iteration methods",
    "author": [{"family": "Polyak", "given": "Boris T."}],
    "container-title": "USSR Computational Mathematics and Mathematical Physics",
    "volume": "4",
    "page": "1-17",
    "issued": {"date-parts": [[1964]]}
  },
  {
    "id": "Nesterov1983",
    "type": "article-journal",
    "title": "A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)",
    "author": [{"family": "Nesterov", "given": "Yurii"}],
    "container-title": "Soviet Mathematics Doklady",
    "volume": "27",
    "page": "372-376",
    "issued": {"date-parts": [[1983]]}
  },
  {
    "id": "KingmaBa2015",
    "type": "paper-conference",
    "title": "Adam: A Method for Stochastic Optimization",
    "author": [{"family": "Kingma", "given": "Diederik P."}, {"family": "Ba", "given": "Jimmy"}],
    "container-title": "International Conference on Learning Representations (ICLR)",
    "issued": {"date-parts": [[2015]]},
    "URL": "https://arxiv.org/abs/1412.6980"
  },
  {
    "id": "Andrychowicz2016",
    "type": "paper-conference",
    "title": "Learning to learn by gradient descent by gradient descent",
    "author": [
      {"family": "Andrychowicz", "given": "Marcin"},
      {"family": "Denil", "given": "Misha"},
      {"family": "Gomez", "given": "Sergio"},
      {"family": "Hoffman", "given": "Matthew"},
      {"family": "Pfau", "given": "David"},
      {"family": "Schaul", "given": "Tom"},
      {"family": "de Freitas", "given": "Nando"}
    ],
    "container-title": "Advances in Neural Information Processing Systems (NeurIPS)",
    "issued": {"date-parts": [[2016]]}
  },
  {
    "id": "Wichrowska2017",
    "type": "paper-conference",
    "title": "Learned Optimizers that Scale and Generalize",
    "author": [
      {"family": "Wichrowska", "given": "Olga"},
      {"family": "Maheswaranathan", "given": "Niru"},
      {"family": "Hoffman", "given": "Matthew W."},
      {"family": "Gomez Colmenarejo", "given": "Sergio"},
      {"family": "Denil", "given": "Misha"},
      {"family": "de Freitas", "given": "Nando"},
      {"family": "Sohl-Dickstein", "given": "Jascha"}
    ],
    "container-title": "International Conference on Machine Learning (ICML)",
    "issued": {"date-parts": [[2017]]}
  },
  {
    "id": "StanleyMiikkulainen2002",
    "type": "article-journal",
    "title": "Evolving neural networks through augmenting topologies",
    "author": [{"family": "Stanley", "given": "Kenneth O."}, {"family": "Miikkulainen", "given": "Risto"}],
    "container-title": "Evolutionary Computation",
    "volume": "10",
    "issue": "2",
    "page": "99-127",
    "issued": {"date-parts": [[2002]]}
  },
  {
    "id": "Holland1975",
    "type": "book",
    "title": "Adaptation in Natural and Artificial Systems",
    "author": [{"family": "Holland", "given": "John H."}],
    "issued": {"date-parts": [[1975]]},
    "publisher": "University of Michigan Press",
    "publisher-place": "Ann Arbor"
  },
  {
    "id": "Salimans2017",
    "type": "article-journal",
    "title": "Evolution Strategies as a Scalable Alternative to Reinforcement Learning",
    "author": [{"family": "Salimans", "given": "Tim"}, {"family": "Ho", "given": "Jonathan"}, {"family": "Chen", "given": "Xiang"}, {"family": "Sutskever", "given": "Ilya"}],
    "container-title": "arXiv preprint arXiv:1703.03864",
    "issued": {"date-parts": [[2017]]},
    "URL": "https://arxiv.org/abs/1703.03864"
  },
  {
    "id": "KirschSchmidhuber2021",
    "type": "paper-conference",
    "title": "Meta Learning Backpropagation and Improving It",
    "author": [{"family": "Kirsch", "given": "Louis"}, {"family": "Schmidhuber", "given": "Jürgen"}],
    "container-title": "Advances in Neural Information Processing Systems (NeurIPS)",
    "issued": {"date-parts": [[2021]]},
    "URL": "https://arxiv.org/abs/2012.14905"
  }
]

import json, pathlib

path_xml = "word_bibliography.xml"
path_bib = "/mnt/data/references.bib"
path_ris = "/mnt/data/references.ris"
path_json = "/mnt/data/references.csl.json"

with open(path_xml, "w", encoding="utf-8") as f:
    f.write(word_xml)

with open(path_bib, "w", encoding="utf-8") as f:
    f.write(bibtex.strip()+"\n")

with open(path_ris, "w", encoding="utf-8") as f:
    f.write(ris.strip()+"\n")

with open(path_json, "w", encoding="utf-8") as f:
    json.dump(csl_json, f, ensure_ascii=False, indent=2)

[path_xml, path_bib, path_ris, path_json]
