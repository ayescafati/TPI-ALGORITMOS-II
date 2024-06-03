import pytest
from c45 import C45

@pytest.fixture
def datos():
    return [
        [1, 'Rojo', 'Si'],
        [1, 'Rojo', 'No'],
        [1, 'Verde', 'Si'],
        [2, 'Rojo', 'No'],
        [2, 'Verde', 'Si'],
        [2, 'Verde', 'Si'],
    ]

@pytest.fixture
def c45(datos):
    return C45(datos)

def test_conteosUnicos(c45, datos):
    conteos = c45.conteosUnicos(datos)
    assert conteos == {'Si': 4, 'No': 2}

def test_entropia(c45, datos):
    entropia = c45.entropia(datos)
    assert round(entropia, 3) == 0.918

def test_dividirConjunto(c45, datos):
    conjunto1, conjunto2 = c45.dividirConjunto(datos, 1, 'Rojo')
    assert len(conjunto1) == 3
    assert len(conjunto2) == 3
