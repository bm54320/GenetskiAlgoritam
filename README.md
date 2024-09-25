Zadatak vjezbe jest napisati program za ucenje (treniranje) umjetne neuronske mreze (tj. odredivanje njezinih tezina) pomocu genetskog algoritma. Neuronska mreza kakva se koristi
u ovom zadatku tipicno se uci (trenira) algoritmom propagacije pogreske unazad (engl.
backpropagation algorithm).

Kod mora primati iduće argumente:

1. Putanja do datoteke skupa podataka za treniranje (--train)
2. Putanja do datoteke skupa podataka za testiranje (--test)
3. Arhitektura neuronske mreˇze (--nn)
4. Veliˇcina populacije genetskog algoritma (--popsize)
5. Elitizam genetskog algoritma (--elitism)
6. Vjerojatnost mutacije svakog elementa kromosoma genetskog algoritma (--p)
7. Standardna devijacija Gaussovog ˇsuma mutacije (--K)
8. Broj iteracija genetskog algoritma (--iter)

Implementacija koda se provjerava pomocu "autograder"-a. To je poseban kod napravljen na kolegiju zvan "Uvod u umjetnu inteligenciju" na 3. godini FER-a.
