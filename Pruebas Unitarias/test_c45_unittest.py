import unittest
from c45 import C45

class TestC45(unittest.TestCase):
    def setUp(self):
        self.datos = [
            [1, 'Rojo', 'Si'],
            [1, 'Rojo', 'No'],
            [1, 'Verde', 'Si'],
            [2, 'Rojo', 'No'],
            [2, 'Verde', 'Si'],
            [2, 'Verde', 'Si'],
        ]
        self.c45 = C45(self.datos)

    def test_conteosUnicos(self):
        conteos = self.c45.conteosUnicos(self.datos)
        self.assertEqual(conteos, {'Si': 4, 'No': 2})

    def test_entropia(self):
        entropia = self.c45.entropia(self.datos)
        self.assertAlmostEqual(entropia, 0.918, places=3)

    def test_dividirConjunto(self):
        conjunto1, conjunto2 = self.c45.dividirConjunto(self.datos, 1, 'Rojo')
        self.assertEqual(len(conjunto1), 3)
        self.assertEqual(len(conjunto2), 3)

if __name__ == '__main__':
    unittest.main()
